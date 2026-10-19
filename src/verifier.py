"""真上カメラの画像から「どのマスにどの色の薬があるか」を読み取る確認モジュール。

判定は色検出（画像処理）で行い、各薬のワールド座標(x,y)まで返す。
GEMINI_API_KEY があるときは Gemini(VLM) にもマスと色を読ませて二重チェックする
（VLMは座標は返さない。取り直しの座標は画像処理側で求める）。
"""
import numpy as np
from planner import DAYS, SLOTS, CELL, CASE_X0, CASE_Y0, cell_of_xy
from config import CONFIG

# 薬の色は config.yaml から（0-1 → 0-255）。色検出の照合に使う。
REF = {k: tuple(int(round(c * 255)) for c in v) for k, v in CONFIG["pills"]["colors"].items()}


def _pixel_to_world(u, v, K, E, z=0.015):
    """画素(u,v)と高さzから、机の上のワールド座標(x,y)を逆算する。

    外部行列 E（world→camera）から一般に計算するので、カメラを動かしても追従する。
    """
    fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
    d_cam = np.array([(u - cx) / fx, (v - cy) / fy, 1.0])   # カメラ座標での視線方向
    E = np.asarray(E, float)
    R, t = E[:3, :3], E[:3, 3]
    Rt = R.T                                                 # カメラ→ワールドの回転
    a = (Rt @ d_cam)[2]
    b = (Rt @ t)[2]
    s = (z + b) / a                                          # ワールドzがzになるスケール
    P = Rt @ (s * d_cam - t)
    return float(P[0]), float(P[1])


def detect_from_image(img, K, E):
    """画像から色つきの塊を検出し、[{color,u,v,x,y}] を返す（画像処理版）。"""
    from collections import deque
    small = img[::2, ::2].astype(np.int16)
    mx, mn = small.max(2), small.min(2)
    sat = (mx - mn) > 55                      # 色がついている画素
    names = list(REF)
    refs = np.array([REF[n] for n in names], dtype=np.int16)
    d = np.linalg.norm(small[..., None, :] - refs[None, None], axis=3)
    cls = d.argmin(2)
    best = d.min(2)
    mask = sat & (best < 90)                  # どの基準色にも近い
    seen = np.zeros(mask.shape, bool)
    out = []
    H, W = mask.shape
    for i in range(H):
        for j in range(W):
            if not mask[i, j] or seen[i, j]:
                continue
            q, pix = deque([(i, j)]), []
            seen[i, j] = True
            while q:
                a, b = q.popleft(); pix.append((a, b))
                for da, db in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    x, y = a + da, b + db
                    if 0 <= x < H and 0 <= y < W and mask[x, y] and not seen[x, y] and cls[x, y] == cls[i, j]:
                        seen[x, y] = True; q.append((x, y))
            if len(pix) < 12:                 # 小さすぎる塊はノイズとして無視
                continue
            a, b = np.mean(pix, 0)
            u, v = b * 2 + 1, a * 2 + 1       # 縮小を戻す
            x, y = _pixel_to_world(u, v, K, E)
            out.append({"color": names[cls[i, j]], "u": float(u), "v": float(v), "x": float(x), "y": float(y)})
    return out


def blob_to_cell(b):
    """検出した塊が入っているマスを返す（ケースの外なら None）。"""
    if b["x"] <= CASE_X0 + 0.005:
        return None
    return cell_of_xy(b["x"], b["y"])


def read_grid(img, K, E):
    """画像→ {マス: 色} と、ケース外の薬リスト。"""
    grid, outside = {}, []
    for b in detect_from_image(img, K, E):
        c = blob_to_cell(b)
        if c: grid[c] = b["color"]
        else: outside.append(b)
    return grid, outside


def compare(plan, grid):
    """作業リストと実際のマスの中身を比べ、ずれを返す。

    wrong: 指定と違う色が入っているマス / missing: 空のマス
    extra: 指定に無いのに薬が入っているマス
    """
    want = {(t["day"], t["slot"]): t["pill"] for t in plan}
    wrong = [(c, want[c], grid.get(c)) for c in want if c in grid and grid[c] != want[c]]
    missing = [c for c in want if c not in grid]
    extra = [c for c in grid if c not in want]
    return wrong, missing, extra


def read_grid_vlm(img, plan=None):
    """Gemini(VLM)で「どのマスにどの色があるか」を読む。返り値は read_grid と同じ辞書。

    座標は返さない（取り直しの座標は画像処理側 detect_from_image で求める）。
    plan（期待する作業リスト）は渡さない: 期待値を教えると「画像を読む」ではなく
    「答えを写す」になり、独立した二重チェックにならないため。
    """
    import os, json
    from google import genai
    import imageio.v3 as iio
    from planner import PILLS, gemini_generate
    import tempfile
    path = os.path.join(tempfile.gettempdir(), "pill_sorter_top_vlm.png")
    iio.imwrite(path, img)
    c = genai.Client(api_key=os.environ["GEMINI_API_KEY"])
    up = c.files.upload(file=path)
    prompt = ("真上から撮った服薬ケースの画像です。各マスにどの色の薬が入っているかを読み取り、"
              'JSONのみで返してください: {"grid": {"mon-am": "red", ...}, "outside": ["red"]}。'
              f"色は{PILLS}。曜日は{DAYS}、時間帯は{SLOTS}(例 mon-am)。"
              "空のマスはキーごと書かないでください（null も入れない）。"
              "ケースの並び: 画像の左から月・火・水・木・金・土・日、上段が朝、下段が夜です。")
    res = gemini_generate([up, prompt], {"response_mime_type": "application/json"})
    return parse_vlm_grid(json.loads(res.text))


# 色名のゆれを吸収する（日本語で返ってきても英語に直す）
COLOR_ALIASES = {"赤": "red", "青": "blue", "黄": "yellow", "黄色": "yellow",
                 "緑": "green", "緑色": "green"}


def parse_vlm_grid(d):
    """VLMのJSONから「どのマスにどの色か」を作る（read_grid と同じ形）。

    モデルは指示に反して、空のマスを null や "empty" のような値で返して
    くることがある。これを色として数えると、色は正しく読めていても画像処理と
    食い違い、二重チェックが無駄に不一致になる。そこで
      (1) 空（null/空文字）は捨てる
      (2) 色名は日本語なら英語に直す
      (3) config の色の一覧に無い値（"empty" など）は捨てる
    の3つで「色として扱える値だけ」を残す。
    """
    from config import CONFIG
    valid = set(CONFIG["pills"]["colors"])
    grid = {}
    for k, v in (d or {}).get("grid", {}).items():
        if not isinstance(v, str) or not v.strip() or "-" not in k:
            continue
        raw = v.strip()
        c = COLOR_ALIASES.get(raw, raw.lower())
        if c not in valid:                     # 色の一覧に無い値は捨てる
            continue
        day, slot = k.split("-", 1)
        if day in DAYS and slot in SLOTS:
            grid[(day, slot)] = c
    return grid
