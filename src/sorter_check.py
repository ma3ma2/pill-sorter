"""ステップ3: 自己確認とやり直し。

わざと失敗を起こし（薬がケースの外に落ちる／違うマスに入る）、
真上カメラの画像からずれを検出して、ロボット自身が取り直せるかを試す。
確認なし・ありでマス正答率を比較する。

実行:
  python src/sorter_check.py [--check] [--gpu] [--vlm] [--no-video] [--seed N] [--random-faults]
"""
import os, sys, json, numpy as np, genesis as gs, imageio
from PIL import Image, ImageDraw, ImageFont
import sorter as S
import verifier as V
import overlay as O
import visualize as G
from textutil import find_font
from config import CONFIG

PLAN = CONFIG["prescription"]["check"]
FAULTS = dict(CONFIG["faults"])    # 仕込む失敗の種類（色 -> outside / wrong）
# ケース外に落とす薬の置き場（重ならないように薬ごとに分ける）
OUTSIDE_SPOTS = [(0.34, 0.13), (0.44, 0.13), (0.34, 0.02), (0.44, 0.02)]
FONT_PATH = find_font()
FONT = ImageFont.truetype(FONT_PATH, 26) if FONT_PATH else ImageFont.load_default()

USE_VLM = ("--vlm" in sys.argv) or (os.environ.get("USE_VLM_CHECK") == "1")


def _arg_int(flag, default):
    if flag in sys.argv:
        try:
            return int(sys.argv[sys.argv.index(flag) + 1])
        except Exception:
            return default
    return default


def read_state(img, tasks, K, E, use_vlm=None):
    """現在のケースの中身を読む。

    判断と座標は画像処理（決定的）で行う。use_vlm（既定は --vlm 指定時）のときは
    Gemini(VLM) にも独立に読ませて二重チェックする（不一致でも判断は画像処理を優先）。
    戻り値: (マスの中身, ケース外の薬, VLMの読み取り or None)
    """
    if use_vlm is None:
        use_vlm = USE_VLM
    grid, outside = V.read_grid(img, K, E)
    vlm = None
    if use_vlm:
        try:
            vlm = V.read_grid_vlm(img, tasks) or None
            if vlm and vlm == grid:
                print("  [VLM] 画像処理と一致（二重チェックOK）")
            elif vlm:
                print("  [VLM] 画像処理と不一致 → 画像処理を採用。VLM:", vlm)
            else:
                print("  [VLM] Geminiの回答を解釈できず → 画像処理を採用")
        except Exception as e:
            print("  [VLM] Gemini失敗→画像処理のみ:", e)
            vlm = None
    return grid, outside, vlm


def ai_read_text(vlm, grid):
    """動画の字幕用に「AI(VLM)の読み取り」を1行にする（VLM不可なら画像処理）。"""
    src = "AI(VLM)" if vlm is not None else "画像処理"
    shown = vlm if vlm is not None else grid
    order = {d: i for i, d in enumerate(S.DAYS)}
    order_s = {x: i for i, x in enumerate(S.SLOTS)}
    items = [f"{d}-{x}={c}" for (d, x), c in
             sorted(shown.items(), key=lambda kv: (order_s[kv[0][1]], order[kv[0][0]]))
             if c]                       # 空(null)は出さない
    base = f"{src}の読み取り: " + (", ".join(items) if items else "空")
    if vlm is not None and vlm == grid:
        base += "（画像処理と一致）"
    elif vlm is not None:
        base += "（画像処理と不一致→画像処理を採用）"
    return base


def subtitle(frame, text):
    """下に黒帯を敷いて字幕を焼き込む。"""
    if not text:
        return frame
    img = Image.fromarray(frame)
    d = ImageDraw.Draw(img, "RGBA")
    w, h = img.size
    d.rectangle([0, h - 58, w, h], fill=(0, 0, 0, 165))
    d.text((16, h - 45), text, font=FONT, fill=(255, 255, 255, 255))
    return np.asarray(img)


def _fault_plan(tasks, rng, random_faults):
    """どの薬に、どんな失敗を仕込むかを決める。"""
    if not random_faults:
        return dict(FAULTS)
    pills = [t["pill"] for t in tasks]
    k = min(2, len(pills))
    chosen = list(rng.choice(pills, size=k, replace=False))
    return {p: str(rng.choice(["outside", "wrong"])) for p in chosen}


def _wrong_cell(tasks, rng, randomize=True):
    """まだ使っていない空きマスを1つ返す（わざと違うマスに入れるため）。

    randomize=False のときは先頭の空きマスを使い、実行ごとに変わらないようにする。
    """
    targets = {(t["day"], t["slot"]) for t in tasks}
    empties = [(d, s) for d in S.DAYS for s in S.SLOTS if (d, s) not in targets]
    if not empties:
        return "thu", "am"
    return empties[int(rng.integers(len(empties)))] if randomize else empties[0]


def _record(mode, tasks, score, grid, outside, seed, random_faults=False):
    """実験結果を out/metrics.json に追記する（レポートの表のもと）。"""
    import datetime
    try:
        os.makedirs("out", exist_ok=True)
        path = "out/metrics.json"
        data = []
        if os.path.exists(path):
            with open(path, encoding="utf-8") as f:
                data = json.load(f)
        data.append({
            "mode": mode,                       # "no_check" or "check"
            "plan": [f"{t['day']}-{t['slot']}={t['pill']}" for t in tasks],
            "score": score, "total": len(tasks), "seed": seed,
            "random_faults": bool(random_faults),
            "grid": {f"{d}-{s}": c for (d, s), c in grid.items()},
            "outside": [b["color"] for b in outside],
            "ts": datetime.datetime.now().isoformat(timespec="seconds"),
        })
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=1)
    except Exception as e:
        print("[metrics] 保存に失敗:", e)


def run(check, video_path=None, seed=None, random_faults=False):
    gs.init(backend=gs.gpu if "--gpu" in sys.argv else gs.cpu, logging_level="warning")
    if seed is not None:
        np.random.seed(seed)                            # 再現性
    rng = np.random.default_rng(seed)
    tasks, _ = S.make_plan(PLAN)
    faults = _fault_plan(tasks, rng, random_faults)
    wrong_target = {t["pill"]: _wrong_cell(tasks, rng, random_faults)
                    for t in tasks if faults.get(t["pill"]) == "wrong"}
    # ケース外に落とす薬は、1つずつ違う場所へ（重なると回収が失敗するため）
    outside_dst = {}
    for t in tasks:
        if faults.get(t["pill"]) == "outside":
            outside_dst[t["pill"]] = OUTSIDE_SPOTS[len(outside_dst) % len(OUTSIDE_SPOTS)]
    scene, franka, pills, cam, top = S.build_scene(tasks)
    K, E = np.asarray(top.intrinsics), np.asarray(top.extrinsics)
    writer = imageio.get_writer(video_path, fps=CONFIG["video"]["fps"]) if video_path else None
    sub = {"text": f"処方: {PLAN}"}

    def frames():
        if writer is None:
            return
        topi = O.draw_targets(O.draw_case_labels(top.render()[0], K, E), tasks, K, E)
        view = np.concatenate([cam.render()[0], topi], axis=1)
        writer.append_data(subtitle(S._pad16(view), sub["text"]))

    bot = S.Robot(scene, franka, frames)
    bot.step(40)
    # --- 仕分け（一部にわざと失敗を仕込む）---
    for t, p in zip(tasks, pills):
        src = p.get_pos().cpu().numpy()[:2]
        target = S.cell_center(t["day"], t["slot"])
        if t["pill"] in faults:
            if faults[t["pill"]] == "outside":
                dst = outside_dst[t["pill"]]   # ケース外（薬ごとに別の場所へ）
                msg = f"[失敗を仕込み] {t['pill']} をわざとケースの外に落とす"
            else:
                dst = S.cell_center(*wrong_target[t["pill"]])            # 違うマスへ
                msg = f"[失敗を仕込み] {t['pill']} をわざと違うマスに入れる"
            sub["text"] = msg; print("  " + msg)
        else:
            dst = target; sub["text"] = f"{t['pill']} を {t['day']}-{t['slot']} へ"
        bot.pick_place(src, dst)
    bot.park()
    sub["text"] = "真上カメラの画像から仕分け結果を確認中…"
    grid, outside, vlm = read_state(top.render()[0], tasks, K, E)   # 初回だけVLMで二重チェック
    sub["text"] = ai_read_text(vlm, grid)
    bot.step(180)          # 読み取り字幕を約3秒見せる
    print("[確認なしの結果]", grid, "ケース外:", [b["color"] for b in outside])
    try:
        G.draw_grid_figure(tasks, grid, "out/figure_before.png", "確認前")
    except Exception as e:
        print("[figure] 保存に失敗:", e)
    if check:
        for r in range(5):
            wrong, missing, extra = V.compare(tasks, grid)
            if not (wrong or missing):
                break
            for t in tasks:
                target = (t["day"], t["slot"])
                if grid.get(target) == t["pill"]:
                    continue
                b = next((b for b in outside if b["color"] == t["pill"]), None)
                if b:
                    src = (b["x"], b["y"]); why = f"{t['pill']} がケースの外 → 回収して入れ直す"
                else:
                    c = next((c for c, col in grid.items() if col == t["pill"]), None)
                    if c is None:
                        continue
                    src = S.cell_center(*c)
                    why = f"{t['pill']} が {c[0]}-{c[1]} → 正しい {target[0]}-{target[1]} へ入れ直す"
                sub["text"] = "ずれを検出: " + why; print("  取り直し:", why)
                bot.goto(*src, S.LIFT_Z); bot.pick_place(src, S.cell_center(*target))
                bot.park()
                grid, outside, vlm = read_state(top.render()[0], tasks, K, E, use_vlm=False)
        sub["text"] = "再確認: すべて正しいマスに収まりました" if not (V.compare(tasks, grid)[0] or V.compare(tasks, grid)[1]) else "再確認: まだずれが残っています"
    bot.step(180)          # 成功の字幕を約3秒見せる
    if writer:
        writer.close()
    if os.environ.get("DEBUG_POS"):
        for t, p in zip(tasks, pills):
            pos = p.get_pos().cpu().numpy()
            print(f"  [pos] {t['pill']:6s} target={t['day']}-{t['slot']}  actual=({pos[0]:.3f},{pos[1]:.3f},{pos[2]:.3f})")
    want = {(t["day"], t["slot"]): t["pill"] for t in tasks}
    score = sum(1 for c, col in want.items() if grid.get(c) == col)   # マス単位で正誤
    mode = "check" if check else "no_check"
    print(f"最終: 確認{'あり' if check else 'なし'} マス正答 {score}/{len(tasks)}  実際={grid} ケース外={[b['color'] for b in outside]}")
    try:
        G.draw_grid_figure(tasks, grid, "out/figure_after.png", "確認後")
    except Exception as e:
        print("[figure] 保存に失敗:", e)
    print("RESULT " + json.dumps({"mode": mode, "score": score, "total": len(tasks), "seed": seed}))
    _record(mode, tasks, score, grid, outside, seed, random_faults)
    return score


if __name__ == "__main__":
    check = "--check" in sys.argv
    no_video = "--no-video" in sys.argv
    vid = ("out/check.mp4" if check else None) if not no_video else None
    run(check, vid, seed=_arg_int("--seed", None), random_faults="--random-faults" in sys.argv)
