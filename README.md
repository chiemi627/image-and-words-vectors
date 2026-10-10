# 画像と言葉を同じベクトルで考える 🖼️

線形代数学の授業・オープンキャンパス用の **マルチモーダル（画像×文字）ベクトル演習アプリ**です。
日本語CLIP **[LINE clip-japanese-base](https://huggingface.co/line-corporation/clip-japanese-base)** を使い、
**画像も言葉も同じベクトル空間**に埋め込んで、次の3つを体験できます。

| タブ | 内容 | 例 |
|---|---|---|
| ① 文から画像をさがす | テキスト → 画像 | 「かわいい犬」→ 柴犬の写真 |
| ② 画像から言葉をさがす | 画像 → 言葉（候補リストから） | 飛行機の写真 → 「飛行機」 |
| ③ 似た画像をさがす | 画像 → 画像 | 1枚選ぶと近い画像が並ぶ |

「近いベクトル＝意味が近い」という考え方は、姉妹アプリ
[**意味をベクトルで考える（単語ベクトル）**](https://github.com/chiemi627/meaning-as-vectors) と同じです。
違いは、片方が **画像のベクトル** になるだけ。

---

## 必要なもの

- **Python 3.9 以上**（Apple Silicon Mac / Linux で動作確認）
- ディスク空き容量 **約 2〜3GB**（PyTorch とモデル）
- セットアップ時と初回起動時にネット接続（ライブラリ・モデル・サンプル画像の取得）

---

## セットアップ（初回のみ）

```bash
git clone https://github.com/chiemi627/image-and-words-vectors.git
cd image-and-words-vectors
bash setup.sh
```

`setup.sh` は次を自動で行います。

1. `venv/` に Python 仮想環境を作成
2. `requirements.txt` のライブラリをインストール（torch など）
3. CC0（著作権フリー）のサンプル画像を `images/` にダウンロード

> モデル本体（日本語CLIP, 数百MB）は **初回起動時に自動ダウンロード**されます。

---

## 起動のしかた

```bash
./venv/bin/streamlit run app.py
```

ブラウザが自動で開きます（開かないときは `http://localhost:8501`）。
macOS なら **`起動.command` をダブルクリック**でもOKです。

### 終了のしかた

- ブラウザのタブを閉じる
- 起動時のターミナルで **Control + C**

---

## 自分の画像を使う

`images/` フォルダに写真（.jpg / .png など）を入れるだけです。
オープンキャンパスなら**キャンパスの写真**を入れておくと盛り上がります。
入れ替えたら、アプリの各タブで検索し直せば自動で反映されます。

「画像→言葉」で出す候補の言葉は `labels.json` で自由に追加・削除できます。

---

## 設定メモ

- **使用モデル**: `line-corporation/clip-japanese-base`（Apache 2.0）
- **transformers は 4.x 系が必須**（5.x だと `trust_remote_code` のモデルコードが動きません。`requirements.txt` で固定済み）
- 既定は **CPU** 実行（確実に動く）。Apple Silicon の GPU を使うなら、起動前に環境変数で切替できます:
  ```bash
  CLIP_DEVICE=mps ./venv/bin/streamlit run app.py
  ```

---

## ファイル構成

```
image-and-words-vectors/
├── app.py                   … アプリ本体（Streamlit）
├── labels.json              … 「画像→言葉」の候補リスト
├── download_sample_images.py… CC0サンプル画像のダウンローダ（Openverse）
├── requirements.txt         … 必要なライブラリ（transformers 4.x 固定）
├── setup.sh                 … 初回セットアップ
├── 起動.command             … macOS 用ワンクリック起動
└── images/                  … 画像（Git には含めず、各自でDL/配置）
```

---

## ライセンス / クレジット

- モデル: **LINE clip-japanese-base** — [Apache License 2.0](https://www.apache.org/licenses/LICENSE-2.0)
  <https://huggingface.co/line-corporation/clip-japanese-base>
- サンプル画像: [Openverse](https://openverse.org/) 経由の **CC0（著作権フリー）** 画像のみ。出典は `images/credits.csv` に記録されます。

作成: 渡辺知恵美（産業情報学科）／ 線形代数学 授業教材
