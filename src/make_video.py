"""提出用の動画（約2分）を組み立てる。

シーンごとに別プロセスで実行し、最後に1本へ連結する。
  python src/make_video.py --seg title    # タイトル・課題設定・処方の理解
  python src/make_video.py --seg sort     # 仕分け（2倍速）
  python src/make_video.py --seg check    # 確認とやり直し（等速）
  python src/make_video.py --seg result   # 結果とまとめ
  python src/make_video.py --seg concat   # out/final.mp4 に連結
"""
import json
import os
import subprocess
import sys

import imageio
import numpy as np
from PIL import Image, ImageDraw, ImageFont

import overlay as O
import planner as P
from config import CONFIG
from textutil import find_font

FPS = CONFIG["video"]["fps"]
W, H = 1440, 544
BG = (16, 22, 30)
FONT_PATH = find_font()
def F(sz): return ImageFont.truetype(FONT_PATH, sz) if FONT_PATH else ImageFont.load_default()
FT, FS, FTag = F(52), F(30), F(30)
OUT = "out"
RX_MAIN = CONFIG["prescription"]["main"]


def blank():
    return np.full((H, W, 3), BG, np.uint8)


def _bg():
    """背景用のシーン静止画（あれば）。無ければ単色。"""
    if os.path.exists(f"{OUT}/bg.png"):
        import imageio.v3 as iio
        b = iio.imread(f"{OUT}/bg.png")
        if b.shape[:2] == (H, W):
            return b
    return blank()


def render_bg():
    """カードの背景にするため、シーンの1コマを描画して保存する。"""
    import genesis as gs

    import sorter as S
    gs.init(backend=gs.gpu if "--gpu" in sys.argv else gs.cpu, logging_level="warning")
    tasks, _ = S.make_plan("赤は月の朝、青は火の夜")
    scene, _franka, _pills, cam, top = S.build_scene(tasks)
    for _ in range(30):
        scene.step()
    frame = S._pad16(np.concatenate([cam.render()[0], top.render()[0]], axis=1))
    import imageio.v3 as iio
    iio.imwrite(f"{OUT}/bg.png", frame)


def card(title, lines, dur, fname, right=None):
    """タイトルと本文の静止カードを動画にする。背景にシーンの1コマを薄く敷く。"""
    img = Image.fromarray(_bg()); d = ImageDraw.Draw(img, "RGBA")
    d.rectangle([0, 0, W, H], fill=(10, 14, 20, 205))   # 背景を暗くして文字を読みやすく
    d.text((60, 70), title, font=FT, fill=(255, 255, 255))
    d.line([60, 150, W - 60, 150], fill=(90, 110, 140), width=2)
    # 行数に応じて行間と文字サイズを縮める（下端で文字が切れないように）
    top, bottom = 186, H - 14
    n = max(1, len(lines))
    gap = min(48, max(26, (bottom - top) // n))
    fs = FS if gap >= 40 else F(26)
    y = top
    for ln in lines:
        col = (230, 230, 230)
        if ln.startswith("# "): ln, col = ln[2:], (120, 200, 255)
        d.text((60, y), ln, font=fs, fill=col); y += gap
    if right is not None:                     # 右側に表などを貼る（暗くしない）
        img.paste(right, (760, 150))
    arr = np.asarray(img)
    w = imageio.get_writer(f"{OUT}/{fname}", fps=FPS)
    for _ in range(int(dur * FPS)):
        w.append_data(arr)
    w.close()


def band(frame, text, speed=False, tag=""):
    """下に字幕、右上に速度タグを焼き込む。"""
    img = Image.fromarray(frame); d = ImageDraw.Draw(img, "RGBA")
    if text:
        d.rectangle([0, H - 60, W, H], fill=(0, 0, 0, 170))
        d.text((20, H - 48), text, font=FS, fill=(255, 255, 255))
    if speed:
        d.rectangle([W - 250, 16, W - 24, 66], fill=(180, 40, 40, 210))
        d.text((W - 235, 24), tag or "2倍速", font=FTag, fill=(255, 255, 255))
    return np.asarray(img)


def plan_lines(rx):
    tasks, how = P.make_plan(rx)
    label = "Gemini(VLM)" if how.startswith("gemini") else "簡易ルール"
    lines = [f"# 処方: {rx}", f"# 変換: {label}", ""]
    for t in tasks[:7]:
        lines.append(f"  {t['pill']:6s} -> {t['day']}-{t['slot']}")
    return lines


def _target_table(rx):
    """処方の目標配置を表にした画像（カードの右側に貼る）。"""
    import visualize as Vz
    tasks, _ = P.make_plan(rx)
    img = Vz.grid_image(tasks, {}, "目標の配置", show_ok=False, show_plan=True)
    w = 640                                  # 左の文章と重ならない大きさに縮める
    return img.resize((w, round(img.height * w / img.width)))


def seg_title():
    render_bg()
    card("間違えない服薬仕分けロボット",
         ["AI(VLM)による指示理解と、自己確認・やり直しで誤配を防ぐ",
          "Physical AI 応用1 最終課題"],
         5, "seg_title.mp4")
    card("課題と目的",
         ["・高齢者の服薬管理、介護・薬局の配薬は誤りが許されない",
          "・自然文の処方をロボットが理解し、曜日×朝夜のケースへ仕分ける",
          "・『一発実行』ではなく、実行→確認→修正の閉ループで信頼性を高める",
          "・無料のGoogle Colabだけで再現できる構成にする"],
         7, "seg_goal.mp4")
    card("処方の理解（AI）", plan_lines(RX_MAIN), 7, "seg_plan.mp4", right=_target_table(RX_MAIN))


def _overlay_writer(fname, speed=False, tag=""):
    w = imageio.get_writer(f"{OUT}/{fname}", fps=FPS)
    def frames(view, text):
        w.append_data(band(view, text, speed, tag))
    return w, frames


def seg_sort():
    import genesis as gs

    import sorter as S
    gs.init(backend=gs.gpu if "--gpu" in sys.argv else gs.cpu, logging_level="warning")
    tasks, _ = S.make_plan(RX_MAIN)
    scene, franka, pills, cam, top = S.build_scene(tasks)
    K, E = np.asarray(top.intrinsics), np.asarray(top.extrinsics)
    w, frames = _overlay_writer("seg_sort.mp4", speed=True, tag="仕分け 2倍速")
    n = {"i": 0}
    def cb():
        n["i"] += 1
        if n["i"] % 2:      # 1フレームおきに書いて2倍速にする
            return
        topi = O.draw_targets(O.draw_case_labels(top.render()[0], K, E), tasks, K, E)
        frames(S._pad16(np.concatenate([cam.render()[0], topi], axis=1)),
               "処方どおりに順番に仕分けます")
    bot = S.Robot(scene, franka, cb)
    bot.step(30)
    for t, p in zip(tasks, pills):
        bot.pick_place(p.get_pos().cpu().numpy()[:2], S.cell_center(t["day"], t["slot"]))
    bot.park(); bot.step(20); w.close()


def seg_check():
    import genesis as gs

    import sorter as S
    import sorter_check as C
    import verifier as V
    gs.init(backend=gs.gpu if "--gpu" in sys.argv else gs.cpu, logging_level="warning")
    tasks, _ = S.make_plan(C.PLAN)
    scene, franka, pills, cam, top = S.build_scene(tasks)
    K, E = np.asarray(top.intrinsics), np.asarray(top.extrinsics)
    sub = {"t": "通常は正しく入ります。今回はわざと失敗を起こします"}
    w = imageio.get_writer(f"{OUT}/seg_check.mp4", fps=FPS)
    def cb():
        topi = O.draw_targets(O.draw_case_labels(top.render()[0], K, E), tasks, K, E)
        w.append_data(C.subtitle(S._pad16(np.concatenate([cam.render()[0], topi], axis=1)), sub["t"]))
    bot = S.Robot(scene, franka, cb)
    bot.step(30)
    for t, p in zip(tasks, pills):
        src = p.get_pos().cpu().numpy()[:2]; target = S.cell_center(t["day"], t["slot"])
        if t["pill"] in C.FAULTS:
            if C.FAULTS[t["pill"]] == "outside":
                dst = C.OUTSIDE_SPOTS[0]; sub["t"] = f"わざと失敗: {t['pill']} をケースの外へ"
            else:
                dst = S.cell_center(*C._wrong_cell(tasks, None, False)); sub["t"] = f"わざと失敗: {t['pill']} を違うマスへ"
        else:
            dst = target; sub["t"] = f"{t['pill']} を {t['day']}-{t['slot']} へ"
        bot.pick_place(src, dst)
    bot.park()
    sub["t"] = "真上カメラの画像で仕分け結果を確認します"
    _use_vlm = "--vlm" in sys.argv          # 動画でVLMの読み取りを見せたいときだけ
    grid, outside, vlm = C.read_state(top.render()[0], tasks, K, E, use_vlm=_use_vlm)
    sub["t"] = C.ai_read_text(vlm, grid)      # AI(VLM)の読み取りを字幕で見せる
    bot.step(180)                             # 約3秒表示
    for r in range(5):
        if not (V.compare(tasks, grid)[0] or V.compare(tasks, grid)[1]):
            break
        for t in tasks:
            target = (t["day"], t["slot"])
            if grid.get(target) == t["pill"]:
                continue
            b = next((b for b in outside if b["color"] == t["pill"]), None)
            if b:
                src = (b["x"], b["y"]); why = f"ずれ検出: {t['pill']} がケース外 → 回収して入れ直す"
            else:
                c = next((c for c, col in grid.items() if col == t["pill"]), None)
                if c is None: continue
                src = S.cell_center(*c); why = f"ずれ検出: {t['pill']} が {c[0]}-{c[1]} → 正しいマスへ"
            sub["t"] = why
            bot.goto(*src, S.LIFT_Z); bot.pick_place(src, S.cell_center(*target)); bot.park()
            grid, outside, vlm = C.read_state(top.render()[0], tasks, K, E, use_vlm=False)
    _w, _m, _e = V.compare(tasks, grid)
    sub["t"] = "再確認: すべて正しいマスに収まりました" if not (_w or _m) else "再確認: まだずれが残っています"
    bot.step(180); w.close()      # 成功の字幕を約3秒見せる


def _result_lines():
    """out/metrics.json の実測値から結果カードの行を作る（無ければ中立の表現）。

    無作為な失敗を仕込む追加実験(--random-faults)の記録は使わず、
    ノートの手順で作った「決定的な実験」の値だけを見る。
    """
    try:
        with open(os.path.join(OUT, "metrics.json"), encoding="utf-8") as f:
            data = json.load(f)
        fixed = [r for r in data if not r.get("random_faults")]
        no = [r for r in fixed if r["mode"] == "no_check"]
        ch = [r for r in fixed if r["mode"] == "check"]
        if no and ch:
            last = ("失敗を検出して自分で直せることを確認した"
                    if ch[-1]["score"] == ch[-1]["total"]
                    else "取り直し後もずれが残った（改善の余地）")
            return [f"・確認なし: マス正答 {no[-1]['score']}/{no[-1]['total']}",
                    f"・確認あり: マス正答 {ch[-1]['score']}/{ch[-1]['total']}",
                    "",
                    last]
    except Exception:  # noqa: BLE001, S110  # 記録が無ければ中立の表示にする
        pass
    return ["・確認なし／確認ありを比較（結果は out/metrics.json に保存）"]


def seg_result():
    card("結果", _result_lines(), 7, "seg_result.mp4")
    card("まとめ", ["・AI(VLM)が自然文の処方を作業リストに変換",
                    "・真上カメラの画像から仕分け結果を確認し、ずれを自動修正",
                    "・判断は画像処理で行い、AI(VLM)は独立した二重チェックに使用",
                    "・すべてGoogle Colab上で完結（自前GPU不要）"], 7, "seg_end.mp4")


def seg_concat():
    segs = ["seg_title.mp4", "seg_goal.mp4", "seg_plan.mp4", "seg_sort.mp4",
            "seg_check.mp4", "seg_result.mp4", "seg_end.mp4"]
    lst = os.path.join(OUT, "list.txt")
    with open(lst, "w") as f:
        f.writelines(f"file '{s}'\n" for s in segs)
    import imageio_ffmpeg
    exe = imageio_ffmpeg.get_ffmpeg_exe()
    subprocess.run([exe, "-y", "-f", "concat", "-safe", "0", "-i", "list.txt",
                    "-c", "copy", "final.mp4"], cwd=OUT, check=True)
    print("out/final.mp4 を作成")


if __name__ == "__main__":
    seg = sys.argv[sys.argv.index("--seg") + 1] if "--seg" in sys.argv else "title"
    os.makedirs(OUT, exist_ok=True)
    {"title": seg_title, "sort": seg_sort, "check": seg_check,
     "result": seg_result, "concat": seg_concat}[seg]()
