# pill-sorter — 服薬仕分けロボット（Genesis / Google Colab）

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/ma3ma2/pill-sorter/blob/main/pill_sorter_colab.ipynb)

AI（VLM）の指示理解と自己確認・やり直しを組み合わせ、机の上の「薬」を
曜日×朝夜のケースへ仕分けるロボットのシミュレーションです。
Physical AI 応用1 最終課題用。

## 構成
```
pill-sorter/
├ README.md
├ LICENSE（MIT）
├ .gitignore
├ requirements.txt
├ config.yaml               # 寸法・色・処方・失敗・モデル・fps・シード
├ src/
│  ├ planner.py             # 処方文 → 作業リスト(JSON)。Gemini(VLM)/ルール両対応＋検証
│  ├ sorter.py              # シーン構築と仕分け（掴む→運ぶ→離す）
│  ├ verifier.py            # 真上カメラ画像から各マスの色と座標を読む（Gemini/画像処理）
│  ├ sorter_check.py        # わざと失敗→確認→取り直しの実験
│  ├ make_video.py          # 提出用動画（約2分）の生成
│  ├ overlay.py             # 真上ビューに曜日/朝夜のラベルを描く
│  ├ visualize.py           # 確認前後の状態を表形式の図に保存
│  ├ textutil.py            # 日本語フォントの探索
│  ├ config.py              # config.yaml の読み込み
│  └ demo_pick.py           # 最小デモ（1個を掴んでケースへ）
├ tests/test_units.py       # 幾何・処方変換・投影の単体テスト（Genesis不要）
├ tools/
│  ├ run_experiments.py     # 確認なし/ありをN回ずつ実行し成功率を集計
│  └ check_deliverable.py   # 配布物の軽い健全性チェック（CI用）
├ .github/workflows/ci.yml  # push ごとに lint・単体テスト・配布物チェック
└ pill_sorter_colab.ipynb   # Colab の入口（リポジトリを clone して実行）
```

## Colab で実行
`pill_sorter_colab.ipynb` を Colab で開き、上から順に実行するだけです
（ノートはリポジトリを取得してスクリプトを実行する薄い入口です）。
Gemini を使う場合は Colab の Secrets に `GEMINI_API_KEY` を登録します。

## ローカルで実行
GPUの無いPCでは、先にCPU版PyTorchを入れると軽くなります（genesis-world が torch を使うため）。
```
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt
python tests/test_units.py                 # 単体テスト（Genesis不要）
python src/sorter.py                       # 仕分け
python src/sorter_check.py --check --vlm   # 確認とやり直し（--vlm でGeminiの二重チェック）
python src/make_video.py --seg title       # 提出用動画（順に title→sort→check→result→concat）
python tools/run_experiments.py --trials 3 # 確認なし/ありの成功率を集計（時間がかかる）
```
環境変数 `GEMINI_API_KEY` を設定すると、処方の変換と画像の確認が Gemini 経由になります。
キーが無い／失敗しても、ルール変換と画像処理で最後まで動きます（フォールバック）。
モデルは自動で選びます（`gemini-3.8-flash` 等を順に試す）。固定は `GEMINI_MODEL`。

## 設定
`config.yaml` に寸法・色・処方・失敗の種類・モデル候補・動画fps・乱数シードをまとめています
（`PILL_SORTER_CONFIG` で別ファイルも指定可）。
結果は `out/metrics.json`（確認あり/なしの正答）・`out/sort_result.json`・`out/summary.json` に保存されます。

## Gemini の無料枠について
無料枠は「モデルごとに1日20リクエスト」程度です。割当超過(429)・廃止(404)・一時混雑(503)のときは
自動で次のモデル／ルール／画像処理へ切り替わります。処方の変換結果は `out/plan_cache.json` に
キャッシュし、同じ処方の再実行では再利用します（`GEMINI_REFRESH=1` で無視）。

## 自動チェック（CI）
`.github/workflows/ci.yml` が push ごとに単体テストと配布物チェックを実行します
（Genesis や GPU は不要、1〜2分）。ruff は参考表示（`continue-on-error`）で、失敗扱いにはしません。
