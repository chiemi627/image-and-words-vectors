# -*- coding: utf-8 -*-
"""
画像と言葉を同じベクトルで考える ― 授業・オープンキャンパス用 マルチモーダル演習アプリ
モデル: LINE clip-japanese-base（日本語CLIP。画像と文字を同じ空間に埋め込む）
機能:  ①テキスト→画像  ②画像→言葉  ③画像→画像
起動:  ./venv/bin/streamlit run app.py
"""
import os
import io
import json
import glob
import numpy as np
import torch
import streamlit as st
from PIL import Image
from transformers import AutoImageProcessor, AutoModel, AutoTokenizer

HF_MODEL = "line-corporation/clip-japanese-base"
BASE_DIR = os.path.dirname(__file__)
IMAGES_DIR = os.path.join(BASE_DIR, "images")
LABELS_PATH = os.path.join(BASE_DIR, "labels.json")
# 既定はCPU（booth で確実に動く）。環境変数で mps/cuda に切替可。
DEVICE = os.environ.get("CLIP_DEVICE", "cpu")
IMG_EXTS = ("*.jpg", "*.jpeg", "*.png", "*.webp", "*.JPG", "*.JPEG", "*.PNG")

st.set_page_config(page_title="画像と言葉をベクトルで考える", page_icon="🖼️", layout="wide")
st.markdown(
    """
    <style>
      html, body, [class*="css"] { font-size: 18px; }
      h1 { font-size: 36px !important; }
      .stTextInput input { font-size: 24px; }
      .word-row { font-size: 24px; font-weight: 700; }
      .score { color: #1C7293; font-weight: 700; }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource(show_spinner="日本語CLIPモデルを読み込み中…（初回のみ少し時間がかかります）")
def load_clip():
    tok = AutoTokenizer.from_pretrained(HF_MODEL, trust_remote_code=True)
    proc = AutoImageProcessor.from_pretrained(HF_MODEL, trust_remote_code=True)
    model = AutoModel.from_pretrained(HF_MODEL, trust_remote_code=True).to(DEVICE).eval()
    return tok, proc, model


tokenizer, processor, model = load_clip()


def embed_texts(words):
    """言葉のリスト → 正規化済みベクトル (N, D)"""
    inputs = tokenizer(list(words)).to(DEVICE)
    with torch.no_grad():
        feats = model.get_text_features(**inputs)
        feats = torch.nn.functional.normalize(feats, dim=-1)
    return feats.cpu().numpy()


def embed_image(img: Image.Image):
    """PIL画像 → 正規化済みベクトル (D,)"""
    inputs = processor(img.convert("RGB"), return_tensors="pt").to(DEVICE)
    with torch.no_grad():
        feats = model.get_image_features(**inputs)
        feats = torch.nn.functional.normalize(feats, dim=-1)
    return feats.cpu().numpy()[0]


def list_images():
    files = []
    for pat in IMG_EXTS:
        files += glob.glob(os.path.join(IMAGES_DIR, pat))
    return sorted(set(files))


@st.cache_data(show_spinner="画像をベクトルに変換中…")
def build_image_index(signature, _model_id):
    """images/ の全画像をベクトル化。signature(= (path,mtime)の並び) が変わると再計算。"""
    paths = [p for p, _ in signature]
    vecs = []
    for p in paths:
        try:
            vecs.append(embed_image(Image.open(p)))
        except Exception:
            vecs.append(np.zeros(model.config.projection_dim if hasattr(model.config, "projection_dim") else 512))
    arr = np.array(vecs) if vecs else np.zeros((0, 1))
    return paths, arr


@st.cache_data(show_spinner="言葉をベクトルに変換中…")
def build_label_index(words):
    return list(words), embed_texts(list(words))


def image_signature():
    return tuple((p, int(os.path.getmtime(p))) for p in list_images())


def load_label_words():
    try:
        with open(LABELS_PATH, encoding="utf-8") as f:
            data = json.load(f)
        return [w for w in data.get("words", []) if isinstance(w, str)]
    except Exception:
        return ["犬", "猫", "寿司", "電車", "花", "山", "本", "車"]


st.title("🖼️ 画像と言葉を同じベクトルで考える")
st.caption("日本語CLIP（LINE clip-japanese-base）で、画像も言葉も同じ空間のベクトルにします。"
           "『近いベクトル＝意味が近い』は単語ベクトルと同じ考え方です。")

paths = list_images()
if not paths:
    st.warning("画像がありません。`images/` フォルダに写真を入れてください"
               "（`python download_sample_images.py` でサンプルを取得できます）。")

tab1, tab2, tab3 = st.tabs(["① 文から画像をさがす", "② 画像から言葉をさがす", "③ 似た画像をさがす"])

# ===== ① テキスト → 画像 =====
with tab1:
    st.subheader("文から画像をさがす")
    st.write("言葉や文を入力すると、意味が近い画像を AI が選びます。")
    query = st.text_input("さがす言葉・文", value="かわいい動物", key="t2i_q")
    topn = st.slider("いくつ表示する？", 1, 12, 6, key="t2i_n")
    if st.button("さがす", type="primary", key="t2i_btn") and paths:
        names, img_vecs = build_image_index(image_signature(), HF_MODEL)
        qv = embed_texts([query])[0]
        sims = img_vecs @ qv
        order = np.argsort(-sims)[:topn]
        cols = st.columns(3)
        for i, idx in enumerate(order):
            with cols[i % 3]:
                st.image(names[idx], use_container_width=True,
                         caption=f"{os.path.basename(names[idx])}　類似度 {sims[idx]:.3f}")

# ===== ② 画像 → 言葉 =====
with tab2:
    st.subheader("画像から言葉をさがす")
    st.write("画像を選ぶと、その画像に近い言葉を候補リストから選びます。")
    up = st.file_uploader("画像をアップロード（または下の一覧から選択）",
                          type=["jpg", "jpeg", "png", "webp"], key="i2t_up")
    chosen = None
    if up is not None:
        chosen = Image.open(io.BytesIO(up.read()))
    elif paths:
        name = st.selectbox("フォルダの画像から選ぶ", [os.path.basename(p) for p in paths], key="i2t_sel")
        chosen = Image.open(os.path.join(IMAGES_DIR, name))
    topn2 = st.slider("いくつ表示する？", 3, 15, 8, key="i2t_n")
    if st.button("言葉をさがす", type="primary", key="i2t_btn") and chosen is not None:
        c1, c2 = st.columns([1, 1])
        with c1:
            st.image(chosen, use_container_width=True)
        with c2:
            words, label_vecs = build_label_index(tuple(load_label_words()))
            iv = embed_image(chosen)
            sims = label_vecs @ iv
            # CLIP流に 100倍してsoftmax → 「この画像は〇〇 92%」の見やすい表示
            logits = 100.0 * sims
            probs = np.exp(logits - logits.max())
            probs = probs / probs.sum()
            order = np.argsort(-probs)[:topn2]
            for idx in order:
                pct = float(probs[idx]) * 100
                st.markdown(f'<div class="word-row">{words[idx]} '
                            f'<span class="score">{pct:.0f}%</span></div>', unsafe_allow_html=True)
                st.progress(max(0, min(100, int(round(pct)))))

# ===== ③ 画像 → 画像 =====
with tab3:
    st.subheader("似た画像をさがす")
    st.write("1枚選ぶと、見た目・意味が近い画像を並べます。")
    if paths:
        name = st.selectbox("基準にする画像", [os.path.basename(p) for p in paths], key="i2i_sel")
        topn3 = st.slider("いくつ表示する？", 1, 11, 5, key="i2i_n")
        if st.button("似た画像をさがす", type="primary", key="i2i_btn"):
            names, img_vecs = build_image_index(image_signature(), HF_MODEL)
            base_idx = names.index(os.path.join(IMAGES_DIR, name))
            sims = img_vecs @ img_vecs[base_idx]
            order = [i for i in np.argsort(-sims) if i != base_idx][:topn3]
            st.image(names[base_idx], width=220, caption="基準の画像")
            cols = st.columns(3)
            for i, idx in enumerate(order):
                with cols[i % 3]:
                    st.image(names[idx], use_container_width=True,
                             caption=f"類似度 {sims[idx]:.3f}")
    else:
        st.info("画像を入れると使えます。")
