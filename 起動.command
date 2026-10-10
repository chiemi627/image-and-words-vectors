#!/bin/bash
# ダブルクリックで起動します（ブラウザが自動で開きます）
cd "$(dirname "$0")"
./venv/bin/streamlit run app.py --browser.gatherUsageStats false
