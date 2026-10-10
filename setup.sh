#!/bin/bash
# 初回セットアップ: 仮想環境の作成 → ライブラリ導入 → サンプル画像のダウンロード
# 使い方:  bash setup.sh
set -e
cd "$(dirname "$0")"

echo "==> Python 仮想環境を作成します (venv/)"
python3 -m venv venv

echo "==> ライブラリをインストールします（torch など。数分かかります）"
./venv/bin/pip install --upgrade pip
./venv/bin/pip install -r requirements.txt

echo "==> サンプル画像（CC0）をダウンロードします"
./venv/bin/python -I download_sample_images.py || echo "（画像DLは任意。images/ に自分の写真を入れてもOK）"

echo ""
echo "✓ セットアップ完了！ 次のコマンドで起動できます:"
echo "    ./venv/bin/streamlit run app.py"
echo "  （macOS なら 起動.command をダブルクリックでもOK）"
echo ""
echo "※ 初回起動時に日本語CLIPモデル（数百MB）を自動ダウンロードします（要ネット）。"
