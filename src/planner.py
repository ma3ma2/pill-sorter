"""処方テキスト → 作業リスト(JSON) への変換。

GEMINI_API_KEY があれば Gemini に変換させ、無ければ簡易ルールで変換する。
出力は [{"pill": "red", "day": "mon", "slot": "am"}, ...] の形。
どちらの経路でも最後に validate() で範囲外の指定を弾く（AIの出力をそのまま信用しない）。
"""
import json
import os

from config import CONFIG

DAYS = ["mon", "tue", "wed", "thu", "fri", "sat", "sun"]
SLOTS = ["am", "pm"]
PILLS = ["red", "blue", "yellow", "green"]

# 寸法は config.yaml から読む（環境が変わってもファイルで調整できる）

_G = CONFIG["geometry"]
CELL = _G["cell"]                 # 1マスの大きさ[m]
CASE_X0, CASE_Y0 = _G["case_x0"], _G["case_y0"]
WALL_H = _G["wall_h"]             # 仕切りの高さ
PILL = _G["pill"]                 # 薬の大きさ

DAYS_JP = ["月", "火", "水", "木", "金", "土", "日"]
SLOTS_JP = {"am": "朝", "pm": "夜"}


def cell_center(day, slot):
    """マスの中心座標。

    真上から見て 月..日 が左から右、AM が上、PM が下 になるように配置する。
    （y が小さいほど画像では左、x が大きいほど画像では上）
    """
    j = DAYS.index(day)
    i = SLOTS.index(slot)
    x = CASE_X0 + (len(SLOTS) - 1 - i + 0.5) * CELL
    y = CASE_Y0 + (len(DAYS) - 1 - j + 0.5) * CELL
    return x, y


def cell_of_xy(x, y):
    """座標(x,y)がどのマスかを返す（ケースの外なら None）。"""
    i = int((x - CASE_X0) // CELL)
    j = int((y - CASE_Y0) // CELL)
    if 0 <= i < len(SLOTS) and 0 <= j < len(DAYS):
        return DAYS[len(DAYS) - 1 - j], SLOTS[len(SLOTS) - 1 - i]
    return None
JP_DAY = {"月": "mon", "火": "tue", "水": "wed", "木": "thu", "金": "fri", "土": "sat", "日": "sun"}
JP_PILL = {"赤": "red", "青": "blue", "黄": "yellow", "緑": "green"}

PROMPT = """あなたは服薬仕分けロボットの計画係です。次の処方を、1錠ずつの作業リストに変換してください。
出力はJSON配列のみ。各要素は {{"pill": 色, "day": 曜日, "slot": 時間帯}}。
色は {pills}、曜日は {days}、時間帯は am(朝) / pm(夜) のいずれか。
処方: {text}"""


LAST_MODEL = None   # 直近で成功したモデル名（診断用）


def _candidate_models():
    """使うモデル候補。GEMINI_MODEL が優先、無ければ新しい flash を順に試す。"""
    env = os.environ.get("GEMINI_MODEL")
    base = list(CONFIG["gemini"]["models"])
    return ([env] + [m for m in base if m != env]) if env else base


# 一時的なエラー（サーバ混雑など）。少し待って再試行する。
TRANSIENT = ("unavailable", "503", "500", "internal", "deadline", "overloaded", "try again")
# このモデルでは不可（廃止・割当超過）。待たずに次の候補へ。
SKIP_MODEL = ("not_found", "no longer available", "not found",
              "resource_exhausted", "429", "quota")


def gemini_generate(contents, config=None, retries=3):
    """Gemini 呼び出し。

    一時エラー(503等)は少し待って数回再試行し、モデル廃止(404)や割当超過(429)は
    次の候補モデルへ切り替える。それでも失敗したら例外を投げ、呼び出し側で
    ルール/画像処理へフォールバックする。
    """
    import time

    from google import genai  # pip install google-genai
    client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])
    last = None
    for model in _candidate_models():
        kw = {"model": model, "contents": contents}
        if config:
            kw["config"] = config
        for attempt in range(retries):
            try:
                res = client.models.generate_content(**kw)
                global LAST_MODEL  # noqa: PLW0603  # ノートが最後に使ったモデル名を表示するため
                LAST_MODEL = model
                return res
            except Exception as e:
                last = e
                msg = str(e).lower()
                if any(k in msg for k in SKIP_MODEL):
                    break                       # 次のモデルへ
                if any(k in msg for k in TRANSIENT) and attempt < retries - 1:
                    time.sleep(2 + 2 * attempt)  # 一時エラー → 少し待って再試行
                    continue
                if any(k in msg for k in TRANSIENT):
                    break                       # 再試行してもダメ → 次のモデルへ
                raise
    raise last


def plan_with_gemini(text):
    res = gemini_generate(PROMPT.format(pills=PILLS, days=DAYS, text=text),
                          {"response_mime_type": "application/json"})
    return json.loads(res.text)


def plan_with_rules(text):
    """「赤は月水金の朝、青は毎日夜」のような文を読む簡易版。

    ・「月曜日」のような表記も「月」として読む（曜日/曜 を先に除去）
    ・薬の色ごとに区切るので「赤は月、水、金の朝」も3日分として読める
    ・時間帯（朝/夜）を書かない場合は朝(am)とみなす
    """
    t = text.replace("曜日", "").replace("曜", "")
    hits = [(i, ch) for i, ch in enumerate(t) if ch in JP_PILL]   # 色の出現位置
    tasks = []
    for n, (i, ch) in enumerate(hits):
        j = hits[n + 1][0] if n + 1 < len(hits) else len(t)
        seg = t[i:j]
        pill = JP_PILL[ch]
        days = DAYS if "毎日" in seg else [v for k, v in JP_DAY.items() if k in seg]
        if not days:
            print(f"[planner] 「{ch}」の曜日を読み取れず除外（色を先に書いてください）")
            continue
        slots = [s for s, kw in (("am", "朝"), ("pm", "夜")) if kw in seg] or ["am"]
        tasks += [{"pill": pill, "day": d, "slot": s} for d in days for s in slots]
    return tasks


def validate(tasks):
    ok = [t for t in tasks if t.get("pill") in PILLS and t.get("day") in DAYS and t.get("slot") in SLOTS]
    dropped = len(tasks) - len(ok)
    if dropped:
        print(f"[planner] 不正な指定を {dropped} 件除外")
    return ok


CACHE_PATH = os.path.join("out", "plan_cache.json")


def _cache_get(key):
    try:
        with open(CACHE_PATH, encoding="utf-8") as f:
            return json.load(f).get(key)
    except Exception:  # noqa: BLE001  # キャッシュは無くても動く
        return None


def _cache_put(key, tasks):
    try:
        os.makedirs(os.path.dirname(CACHE_PATH), exist_ok=True)
        d = {}
        if os.path.exists(CACHE_PATH):
            with open(CACHE_PATH, encoding="utf-8") as f:
                d = json.load(f)
        d[key] = tasks
        with open(CACHE_PATH, "w", encoding="utf-8") as f:
            json.dump(d, f, ensure_ascii=False)
    except Exception:  # noqa: BLE001, S110  # キャッシュ保存は失敗してもよい
        pass


def make_plan(text):
    """処方を作業リストに変換する。

    一度Geminiで変換した結果は out/plan_cache.json に保存し、同じ処方なら再利用する
    （無料枠の消費を抑える）。GEMINI_REFRESH=1 でキャッシュを無視して再変換。
    """
    import hashlib
    key = hashlib.md5(text.encode("utf-8")).hexdigest()
    if os.environ.get("GEMINI_API_KEY") and not os.environ.get("GEMINI_REFRESH"):
        hit = _cache_get(key)
        if hit:
            return validate(hit), "gemini(cache)"
    if os.environ.get("GEMINI_API_KEY"):
        try:
            tasks = validate(plan_with_gemini(text))
            _cache_put(key, tasks)
            return tasks, "gemini"
        except Exception as e:  # noqa: BLE001  # 失敗したらルール変換に切り替える
            print("[planner] Gemini失敗、ルールに切替:", e)
    return validate(plan_with_rules(text)), "rules"


if __name__ == "__main__":
    print(make_plan("赤は月と水の朝、青は火の夜"))
