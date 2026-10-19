"""ケースの状態を表形式の図にして保存する（レポート用）。

曜日=列、朝/夜=行。各マスに入っている色を丸で示し、期待と一致すれば OK、
違えば NG を表示する。matplotlib は使わず PIL だけで描く。
"""
from PIL import Image, ImageDraw, ImageFont
from textutil import find_font
from planner import DAYS, SLOTS, DAYS_JP, SLOTS_JP
from config import CONFIG

# 色は config.yaml から（0-1 → 0-255）。設定を変えると図も追従する。
RGB = {k: tuple(int(round(c * 255)) for c in v) for k, v in CONFIG["pills"]["colors"].items()}


def grid_image(plan, grid, title="", show_ok=True, show_plan=False):
    """曜日×朝夜の表を描いて PIL 画像で返す（保存はしない）。

    grid に入っている色を丸で示す。show_plan=True のときは grid の代わりに
    plan（処方で指定されたマス）に色の丸を置く（「目標の配置」の表）。
    show_ok=True のときだけ、plan と一致するかの OK/NG を出す。
    """
    cw, chh = 78, 78
    left, top = 84, 84
    W = left + cw * len(DAYS) + 20
    H = top + chh * len(SLOTS) + 40
    img = Image.new("RGB", (W, H), (255, 255, 255))
    d = ImageDraw.Draw(img)
    fm = ImageFont.truetype(find_font(), 26) if find_font() else ImageFont.load_default()
    fsm = ImageFont.truetype(find_font(), 16) if find_font() else ImageFont.load_default()
    want = {(t["day"], t["slot"]): t["pill"] for t in plan}

    d.text((10, 10), title, font=fm, fill=(0, 0, 0))
    for j, day in enumerate(DAYS):                      # 曜日（列）
        d.text((left + cw * j + cw / 2 - 14, top - 40), DAYS_JP[j], font=fm, fill=(20, 20, 20))
    for i, slot in enumerate(SLOTS):                    # 朝/夜（行）
        d.text((12, top + chh * i + chh / 2 - 16), SLOTS_JP[slot], font=fm, fill=(20, 20, 20))

    for i, slot in enumerate(SLOTS):
        for j, day in enumerate(DAYS):
            x0, y0 = left + cw * j, top + chh * i
            d.rectangle([x0, y0, x0 + cw, y0 + chh], outline=(190, 190, 190))
            cell = (day, slot)
            actual = grid.get(cell)
            shown = want.get(cell) if show_plan else actual
            if shown:
                cx, cy = x0 + cw / 2, y0 + chh / 2 - 6
                d.ellipse([cx - 18, cy - 18, cx + 18, cy + 18], fill=RGB.get(shown, (120, 120, 120)))
            if show_ok and cell in want:
                ok = (actual == want[cell])
                d.text((x0 + 8, y0 + chh - 24), "OK" if ok else "NG", font=fsm,
                       fill=(0, 150, 0) if ok else (205, 0, 0))
    return img


def draw_grid_figure(plan, grid, path, title=""):
    grid_image(plan, grid, title, show_ok=True).save(path)
    return path


def draw_plan_figure(plan, path, title="目標の配置"):
    """処方で指定されたマスに色の丸だけを置いた表（判定は出さない）。"""
    grid_image(plan, {}, title, show_ok=False, show_plan=True).save(path)
    return path
