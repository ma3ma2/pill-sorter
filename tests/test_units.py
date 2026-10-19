"""Genesis や GPU が無くても動く単体テスト。

pytest でも `python tests/test_units.py` でも実行できる。
幾何計算・処方変換・投影など、壊れやすい部分を確認する。
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
os.environ.pop("GEMINI_API_KEY", None)      # ルール経路を使う（外部依存なし）

import numpy as np

import config
import overlay
import planner
from planner import DAYS, SLOTS

K = np.array([[741.82, 0.0, 240.0], [0.0, 741.82, 270.0], [0.0, 0.0, 1.0]])
E = np.array([[0, -1, 0, 0], [-1, 0, 0, 0.64], [0, 0, -1, 0.9], [0, 0, 0, 1.0]])


def test_cell_roundtrip():
    for d in DAYS:
        for s in SLOTS:
            x, y = planner.cell_center(d, s)
            assert planner.cell_of_xy(x, y) == (d, s), (d, s)


def test_cell_outside_is_none():
    assert planner.cell_of_xy(-1.0, -1.0) is None
    assert planner.cell_of_xy(2.0, 2.0) is None


def test_rules_plan_basic():
    tasks, how = planner.make_plan("赤は月の朝、青は火の夜")
    assert how == "rules"
    assert {"pill": "red", "day": "mon", "slot": "am"} in tasks
    assert {"pill": "blue", "day": "tue", "slot": "pm"} in tasks


def test_rules_plan_daily():
    tasks, _ = planner.make_plan("青は毎日の夜")
    assert len(tasks) == len(DAYS)
    assert all(t["slot"] == "pm" and t["pill"] == "blue" for t in tasks)


def test_rules_day_list():
    tasks, _ = planner.make_plan("赤は月、水、金の朝")
    assert {(t["day"], t["slot"]) for t in tasks} == {("mon", "am"), ("wed", "am"), ("fri", "am")}


def test_rules_weekday_form():
    tasks, _ = planner.make_plan("赤は月曜日の朝")
    assert {(t["day"], t["slot"]) for t in tasks} == {("mon", "am")}   # 「日」を日曜と誤読しない


def test_rules_default_am():
    tasks, _ = planner.make_plan("青は火")
    assert {(t["day"], t["slot"]) for t in tasks} == {("tue", "am")}   # 時間帯の省略は朝


def test_pixel_world_roundtrip():
    import verifier
    p = (0.61, 0.10, 0.015)
    c = (E @ np.append(p, 1.0))[:3]
    uv = K @ (c / c[2])
    x, y = verifier._pixel_to_world(uv[0], uv[1], K, E, z=0.015)
    assert abs(x - p[0]) < 1e-6 and abs(y - p[1]) < 1e-6


def test_validate_drops_invalid():
    bad = [{"pill": "purple", "day": "mon", "slot": "am"},
           {"pill": "red", "day": "monday", "slot": "am"}]
    assert planner.validate(bad) == []


def test_projection_principal_point():
    u, v = overlay._project((0.64, 0.0, 0.0), K, E)
    assert abs(u - 240.0) < 1e-6 and abs(v - 270.0) < 1e-6


def test_parse_vlm_grid_drops_null_cells():
    import verifier
    # Gemini が空のマスまで null で返してきた実際の形（ログより）
    d = {"grid": {"mon-am": None, "mon-pm": "yellow", "tue-am": None, "tue-pm": "blue",
                  "wed-am": None, "sun-pm": None, "": "red"}, "outside": ["red"]}
    assert verifier.parse_vlm_grid(d) == {("mon", "pm"): "yellow", ("tue", "pm"): "blue"}
    assert verifier.parse_vlm_grid({}) == {}


def test_parse_vlm_grid_drops_non_colors():
    import verifier
    # "empty" のような色でない値、"purple" のような一覧外、"青" のような日本語
    d = {"grid": {"mon-am": "empty", "mon-pm": "yellow", "tue-pm": "青",
                  "wed-am": "purple", "thu-pm": "RED"}}
    assert verifier.parse_vlm_grid(d) == {("mon", "pm"): "yellow",
                                          ("tue", "pm"): "blue",
                                          ("thu", "pm"): "red"}


def test_draw_targets_marks_cells():
    tasks = [{"pill": "red", "day": "mon", "slot": "am"},
             {"pill": "blue", "day": "tue", "slot": "pm"}]
    img = np.zeros((540, 480, 3), np.uint8)
    out = overlay.draw_targets(img, tasks, K, E)
    assert out.shape == img.shape                       # 形は変わらない
    u, v = overlay._project((*planner.cell_center("mon", "am"), 0.0), K, E)
    x0, y0 = round(u - 20), round(v - 20)
    assert not np.array_equal(out[y0:y0 + 5, x0:x0 + 5], img[y0:y0 + 5, x0:x0 + 5])  # 枠が描かれた
    cu, cv = round(u), round(v)
    assert np.array_equal(out[cv, cu], img[cv, cu])     # 中抜き（塗りつぶさない）


def test_config_defaults_have_robot_heights():
    cfg = config.load("存在しないファイル.yaml")          # yaml が読めない環境の既定値
    for k in ["grip_z", "lift_z", "place_z"]:
        assert k in cfg["robot"], k
    assert cfg["robot"]["place_z"] >= cfg["robot"]["grip_z"]   # 床に押しつけない下限


def test_config_sections():
    for key in ["geometry", "pills", "prescription", "faults", "gemini", "robot", "video"]:
        assert key in config.CONFIG


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    for fn in fns:
        fn()
        print("ok:", fn.__name__)
    print(f"{len(fns)} tests passed")
