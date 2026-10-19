"""設定の読み込み。

config.yaml（無ければ config.json）を読み、既定値とマージして返す。
環境変数 PILL_SORTER_CONFIG で別のファイルを指定できる。
PyYAML が無い環境でも、JSON 形式なら読める（無ければ既定値で動く）。
"""
import json
import os

DEFAULTS = {
    "geometry": {"cell": 0.07, "case_x0": 0.58, "case_y0": -0.245, "wall_h": 0.008, "pill": 0.03},
    "pills": {"colors": {"red": [0.9, 0.1, 0.1], "blue": [0.1, 0.3, 0.95],
                          "yellow": [0.95, 0.85, 0.1], "green": [0.1, 0.7, 0.2]}},
    "prescription": {"main": "赤は月と水と金の朝、青は火と木の夜、黄は土の朝",
                     "check": "赤は月の朝、青は火の夜、黄は水の朝"},
    "faults": {"red": "outside", "yellow": "wrong"},
    "gemini": {"models": ["gemini-3.8-flash", "gemini-3.5-flash", "gemini-3.5-flash-lite",
                           "gemini-2.5-flash", "gemini-2.0-flash"]},
    "robot": {"grip_z": 0.13, "lift_z": 0.28, "place_z": 0.14},
    "video": {"fps": 30, "seed": 0},
}


def _merge(dst, src):
    for k, v in (src or {}).items():
        if isinstance(v, dict) and isinstance(dst.get(k), dict):
            _merge(dst[k], v)
        else:
            dst[k] = v
    return dst


def load(path=None):
    path = path or os.environ.get("PILL_SORTER_CONFIG", "config.yaml")
    cfg = json.loads(json.dumps(DEFAULTS))          # 既定の深いコピー
    if os.path.exists(path):
        data = None
        try:
            import yaml  # PyYAML があれば
            with open(path, encoding="utf-8") as f:
                data = yaml.safe_load(f)
        except Exception:  # noqa: BLE001  # yaml/json のどちらでも読み、無ければ既定値を使う
            try:
                with open(path, encoding="utf-8") as f:
                    data = json.load(f)
            except Exception:  # noqa: BLE001  # 同上（読めなければ既定値）
                data = None
        if isinstance(data, dict):
            _merge(cfg, data)
    return cfg


CONFIG = load()
