"""日本語フォントの探索。

Colab では Noto CJK が標準で入っていないことがあるため、
複数の候補パスを広く探し、見つからなければ None を返す。
None の場合は Notebook 側で `apt-get install -y fonts-noto-cjk` を実行する。
"""
import glob

_PATTERNS = [
    # Linux (Colab など)
    "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
    "/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc",
    "/usr/share/fonts/truetype/noto/NotoSansCJK-Regular.ttc",
    "/usr/share/fonts/**/NotoSansCJK*.ttc",
    "/usr/share/fonts/**/NotoSansJP*.otf",
    "/usr/share/fonts/**/*CJK*.tt[cf]",
    "/usr/share/fonts/**/ipa*g*.ttf",
    "/usr/share/fonts/**/*Gothic*.ttf",
    # Windows
    "C:/Windows/Fonts/meiryo.ttc",
    "C:/Windows/Fonts/YuGothM.ttc",
    "C:/Windows/Fonts/msgothic.ttc",
    "C:/Windows/Fonts/NotoSansJP*.otf",
    # macOS
    "/System/Library/Fonts/ヒラギノ角ゴシック W3.ttc",
    "/System/Library/Fonts/Hiragino Sans GB.ttc",
]


def find_font():
    for pat in _PATTERNS:
        hits = sorted(glob.glob(pat, recursive=True))
        if hits:
            return hits[0]
    return None
