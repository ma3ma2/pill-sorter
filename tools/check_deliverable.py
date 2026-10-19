"""CI用の軽い健全性チェック（Genesis・GPU・APIキー不要）。

確認すること:
- 必要なソース・設定ファイルがそろっていること
- すべての Python ファイルが構文エラーなくコンパイルできること
- 処方→作業リスト変換(ルール経路)が期待どおり動くこと
"""
import ast
import json
import os
import pathlib
import sys

root = pathlib.Path(__file__).resolve().parent.parent
src = root / "src"

need = ["planner.py", "sorter.py", "verifier.py", "sorter_check.py", "make_video.py",
        "overlay.py", "visualize.py", "config.py"]
missing = [n for n in need if not (src / n).exists()]
assert not missing, f"src/ に足りないファイル: {missing}"
for extra in ["config.yaml", "requirements.txt", "tests/test_units.py",
              "tools/run_experiments.py"]:
    assert (root / extra).exists(), f"足りないファイル: {extra}"

targets = list(src.glob("*.py")) + [root / "tests" / "test_units.py",
                                    root / "tools" / "run_experiments.py",
                                    root / "tools" / "check_deliverable.py"]
for p in targets:
    ast.parse(p.read_text(encoding="utf-8"))       # 構文エラーがあれば例外

# ノートブックが正しいJSONで、cloneする入口になっていること
nb = json.loads((root / "pill_sorter_colab.ipynb").read_text(encoding="utf-8"))
assert nb.get("nbformat") == 4, "ipynb の形式が不正"
assert any("git clone" in "".join(c["source"]) for c in nb["cells"]), "ノートに clone 手順がありません"

os.chdir(root)                                     # config.yaml を読めるように
os.environ.pop("GEMINI_API_KEY", None)             # ルール経路を強制
sys.path.insert(0, str(src))
import planner  # noqa: E402  # sys.path を通してから読み込む

tasks, how = planner.make_plan("赤は月の朝、青は火の夜")
assert how == "rules", f"想定外の変換経路: {how}"
assert {"pill": "red", "day": "mon", "slot": "am"} in tasks
assert {"pill": "blue", "day": "tue", "slot": "pm"} in tasks
print(f"OK: ファイル構成・全ソースの構文・処方変換を確認しました（{len(targets)}ファイル）")
print("    モデル候補:", planner._candidate_models())
