# -*- coding: utf-8 -*-
"""
サンプル画像をダウンロードして images/ に保存します。
画像は Openverse 経由の CC0（著作権フリー）のものだけを取得し、
出典を images/credits.csv に記録します。

使い方:  ./venv/bin/python download_sample_images.py
"""
import os
import csv
import time
import requests

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
IMAGES_DIR = os.path.join(BASE_DIR, "images")
API = "https://api.openverse.org/v1/images/"
UA = {"User-Agent": "clip-japanese-demo/1.0 (education; tsukuba-tech)"}

# boothで分かりやすい被写体（検索語）。増減自由。
CATEGORIES = [
    "柴犬", "猫", "寿司", "ラーメン", "ケーキ", "コーヒー",
    "電車", "自転車", "飛行機", "桜", "富士山", "海",
    "花", "鳥", "ギター", "サッカーボール", "本", "お城",
]


def fetch_one(query):
    """CC0画像を1件探してダウンロード。 (content, meta) or None"""
    try:
        r = requests.get(API, headers=UA, timeout=30,
                         params={"q": query, "license": "cc0", "page_size": 5})
        r.raise_for_status()
        results = r.json().get("results", [])
    except Exception as e:
        print(f"  検索失敗 ({query}): {e}")
        return None
    for item in results:
        url = item.get("url")
        if not url:
            continue
        try:
            img = requests.get(url, headers=UA, timeout=30)
            img.raise_for_status()
            if len(img.content) < 3000:  # 壊れ/極小はスキップ
                continue
            return img.content, item
        except Exception:
            continue
    return None


def main():
    os.makedirs(IMAGES_DIR, exist_ok=True)
    rows = []
    for i, cat in enumerate(CATEGORIES, 1):
        got = fetch_one(cat)
        if not got:
            print(f"[{i:02d}] {cat}: 取得できませんでした")
            continue
        content, meta = got
        fname = f"{i:02d}_{cat}.jpg"
        with open(os.path.join(IMAGES_DIR, fname), "wb") as f:
            f.write(content)
        rows.append([fname, cat, meta.get("title", ""), meta.get("creator", ""),
                     meta.get("license", ""), meta.get("foreign_landing_url", meta.get("url", ""))])
        print(f"[{i:02d}] {cat}: 保存 {fname}")
        time.sleep(0.5)  # API への配慮

    # 出典の記録（CC0なので表示義務はないが、念のため残す）
    with open(os.path.join(IMAGES_DIR, "credits.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["file", "query", "title", "creator", "license", "source"])
        w.writerows(rows)
    print(f"\n✓ 完了：{len(rows)} 枚を images/ に保存しました（出典: images/credits.csv）")


if __name__ == "__main__":
    main()
