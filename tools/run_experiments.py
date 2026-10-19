"""確認なし / 確認あり の実験を N 回ずつ自動実行し、成功率を集計する。

  python tools/run_experiments.py --trials 3
結果は out/summary.json に保存し、Markdown の表を標準出力に出す。
シミュレーションを毎回走らせるため時間がかかる（目安: 1試行 1〜3 分）。
"""
import argparse, json, os, re, statistics, subprocess, sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def run_one(check, seed, gpu):
    args = [sys.executable, "src/sorter_check.py", "--no-video", "--random-faults",
            "--seed", str(seed)]
    if check:
        args.append("--check")
    if gpu:
        args.append("--gpu")
    env = dict(os.environ, PYTHONPATH="src", PYOPENGL_PLATFORM="osmesa")
    p = subprocess.run(args, cwd=ROOT, env=env, capture_output=True, text=True)
    m = re.search(r"RESULT (\{.*\})", p.stdout)
    if not m:
        print("失敗した試行の出力:", (p.stdout or "")[-400:], (p.stderr or "")[-400:])
        return None
    return json.loads(m.group(1))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--trials", type=int, default=3, help="各条件の試行回数")
    ap.add_argument("--gpu", action="store_true")
    a = ap.parse_args()

    summary = {}
    for mode, check in [("no_check", False), ("check", True)]:
        acc = []
        for i in range(a.trials):
            r = run_one(check, i, a.gpu)
            if r:
                acc.append(r["score"] / r["total"])
                print(f"  {mode} trial {i}: {r['score']}/{r['total']}")
        summary[mode] = {
            "trials": len(acc),
            "mean_accuracy": round(statistics.mean(acc), 3) if acc else None,
            "scores": acc,
        }

    os.makedirs(os.path.join(ROOT, "out"), exist_ok=True)
    with open(os.path.join(ROOT, "out", "summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=1)

    print("| 条件 | 試行数 | 平均正答率 |")
    print("|---|---|---|")
    for mode, label in [("no_check", "確認なし"), ("check", "確認あり")]:
        s = summary[mode]
        print(f"| {label} | {s['trials']} | {s['mean_accuracy']} |")
    print("→ out/summary.json に保存しました")


if __name__ == "__main__":
    main()
