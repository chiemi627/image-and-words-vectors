# -*- coding: utf-8 -*-
"""
サンプル画像をまとめてダウンロードして images/ に保存します。
画像は Openverse 経由の CC0（著作権フリー）のものだけを取得し、
出典を images/credits.csv に記録します。

使い方:  ./venv/bin/python download_sample_images.py
         ./venv/bin/python download_sample_images.py --per 3 --max 200
"""
import os
import io
import csv
import time
import argparse
import requests
from PIL import Image

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
IMAGES_DIR = os.path.join(BASE_DIR, "images")
API = "https://api.openverse.org/v1/images/"
UA = {"User-Agent": "clip-japanese-demo/1.0 (education; tsukuba-tech)"}

# boothで分かりやすい被写体（検索語）。増減自由。
# 形容詞クエリ（爽やか/美味しい/獰猛/勇ましい/パリパリ…）でも何か当たるよう、
# 動物・食べ物・飲み物・乗り物・自然・建物・もの・人を幅広く入れている。
CATEGORIES = [
    # 動物
    "柴犬", "子犬", "猫", "子猫", "ライオン", "トラ", "ヒグマ", "オオカミ", "ワシ", "タカ",
    "ウマ", "ゾウ", "キリン", "パンダ", "ペンギン", "イルカ", "サメ", "ヘビ", "カエル", "ウサギ",
    "キツネ", "フクロウ", "リス", "ハリネズミ",
    # 食べ物
    "寿司", "ラーメン", "カレーライス", "ピザ", "ハンバーガー", "ショートケーキ", "アイスクリーム",
    "ポテトチップス", "せんべい", "天ぷら", "サラダ", "いちご", "りんご", "バナナ", "ぶどう", "パン",
    # 飲み物
    "コーヒー", "日本茶", "ビール", "ジュース",
    # 乗り物
    "電車", "新幹線", "自動車", "スポーツカー", "自転車", "オートバイ", "飛行機", "ヘリコプター",
    "ヨット", "ロケット", "バス",
    # 自然・風景
    "桜", "富士山", "海", "滝", "青空", "入道雲", "花畑", "ひまわり", "紅葉", "雪山",
    "森", "川", "虹", "星空", "夜景", "花火", "砂漠", "氷山",
    # 建物・場所
    "お城", "神社", "東京タワー", "図書館", "教室", "橋",
    # もの
    "ギター", "ピアノ", "本", "サッカーボール", "傘", "靴", "帽子",
    # 人
    "赤ちゃん", "子ども", "家族", "武士", "笑顔",
    # 画像が豊富で当たりやすい一般語（枚数を稼ぐ）
    "風景", "山", "空", "雲", "湖", "公園", "庭", "道路", "街", "建物",
    "料理", "果物", "野菜", "魚", "木", "植物", "水", "雨", "太陽", "月",
    "女性", "男性", "動物", "犬", "鳥", "海岸", "田んぼ", "都市", "椅子",
]


def fetch_urls(query, want):
    """CC0画像のURLを最大 want 件、Openverse から取得。"""
    try:
        r = requests.get(API, headers=UA, timeout=30,
                         params={"q": query, "license": "cc0", "page_size": max(want * 4, 12)})
        r.raise_for_status()
        results = r.json().get("results", [])
    except Exception as e:
        print(f"  検索失敗 ({query}): {e}")
        return []
    return results


def download(url):
    try:
        img = requests.get(url, headers=UA, timeout=30)
        img.raise_for_status()
        if len(img.content) < 3000:
            return None
        # 中身が本当に画像か検証（拡張子だけ.jpgの壊れファイルを除外）
        Image.open(io.BytesIO(img.content)).verify()
        return img.content
    except Exception:
        return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--per", type=int, default=3, help="1カテゴリあたりの枚数")
    ap.add_argument("--max", type=int, default=200, help="合計の上限枚数")
    args = ap.parse_args()

    os.makedirs(IMAGES_DIR, exist_ok=True)
    seen_urls = set()
    rows = []
    total = 0
    for i, cat in enumerate(CATEGORIES, 1):
        if total >= args.max:
            break
        got = 0
        for item in fetch_urls(cat, args.per):
            if got >= args.per or total >= args.max:
                break
            url = item.get("url")
            if not url or url in seen_urls:
                continue
            seen_urls.add(url)
            content = download(url)
            if not content:
                continue
            got += 1
            total += 1
            fname = f"{i:03d}_{cat}_{got}.jpg"
            with open(os.path.join(IMAGES_DIR, fname), "wb") as f:
                f.write(content)
            rows.append([fname, cat, item.get("title", ""), item.get("creator", ""),
                         item.get("license", ""),
                         item.get("foreign_landing_url", item.get("url", ""))])
        print(f"[{i:03d}] {cat}: {got}枚 （累計 {total}）")
        time.sleep(0.3)

    with open(os.path.join(IMAGES_DIR, "credits.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["file", "query", "title", "creator", "license", "source"])
        w.writerows(rows)
    print(f"\n✓ 完了：{total} 枚を images/ に保存しました（出典: images/credits.csv）")


if __name__ == "__main__":
    main()
