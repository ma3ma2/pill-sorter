"""真上カメラ画像に、曜日(上)と 朝/夜(左) のラベルを描く。

各マスの中心をカメラの投影で画素に変換し、その位置に文字を重ねる。
これで「どのマスが何曜の朝/夜か」が動画で分かる。
"""
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from config import CONFIG
from planner import CASE_X0, CASE_Y0, CELL, DAYS, DAYS_JP, SLOTS, SLOTS_JP, cell_center
from textutil import find_font

# 薬の色（config.yaml の pills.colors、0-1 → 0-255）
RGB = {k: tuple(round(c * 255) for c in v) for k, v in CONFIG["pills"]["colors"].items()}


def _font(sz):
    p = find_font()
    return ImageFont.truetype(p, sz) if p else ImageFont.load_default()


def _project(p, K, E):
    c = (np.asarray(E, float) @ np.append(np.asarray(p, float), 1.0))[:3]
    uv = np.asarray(K, float) @ (c / c[2])
    return float(uv[0]), float(uv[1])


def _text(d, uv, s, f, pad=6, fill=(255, 240, 120)):
    x0, y0, x1, y1 = d.textbbox((0, 0), s, font=f)
    w, h = x1 - x0, y1 - y0
    cx, cy = uv[0] - w / 2 - x0, uv[1] - h / 2 - y0
    d.rectangle([cx - pad, cy - pad, cx + w + pad, cy + h + pad], fill=(0, 0, 0, 150))
    d.text((cx, cy), s, font=f, fill=fill)


def draw_case_labels(frame, K, E, size=26):
    """frame(真上カメラ画像)の周囲に曜日と時間帯のラベルを描いて返す。"""
    img = Image.fromarray(frame)
    d = ImageDraw.Draw(img, "RGBA")
    f = _font(size)
    x_top = CASE_X0 + len(SLOTS) * CELL + 0.025     # ケースより外（画像では上側）
    y_left = CASE_Y0 + len(DAYS) * CELL + 0.025     # ケースより外（画像では左側）
    for day, jp in zip(DAYS, DAYS_JP):              # 曜日（各列の中心の上）
        _, y = cell_center(day, SLOTS[0])
        _text(d, _project((x_top, y, 0.0), K, E), jp, f)
    for slot in SLOTS:                              # 朝/夜（各行の中心の左）
        x, _ = cell_center(DAYS[0], slot)
        _text(d, _project((x, y_left, 0.0), K, E), SLOTS_JP[slot], f)
    return np.asarray(img)


def draw_targets(frame, tasks, K, E, size=40, legend="□ = 処方の位置"):
    """処方どおりに入れるべきマスを、薬の色の中抜き枠で重ねて返す。

    薬（約25px）が枠の中に収まる大きさにし、塗りつぶさない（中が見えるように）。
    重要: 判断に使う画像（read_state / read_grid / read_grid_vlm に渡す生画像）には
    描かないこと。枠を薬と誤検出したり、VLMに答えを見せたりする原因になる。
    """
    img = Image.fromarray(frame)
    d = ImageDraw.Draw(img, "RGBA")
    half = size / 2
    for t in tasks:
        x, y = cell_center(t["day"], t["slot"])
        u, v = _project((x, y, 0.0), K, E)
        col = RGB.get(t["pill"], (200, 200, 200))
        d.rectangle([u - half, v - half, u + half, v + half], outline=col + (255,), width=3)
    if legend:
        _text(d, (120, 26), legend, _font(20), pad=4)
    return np.asarray(img)
