import streamlit as st
import pandas as pd
import os
import pickle
import numpy as np
from PIL import Image
import time

from Helpers.getEmbeddings import getEmbeddings

# ===================== PAGE CONFIG ===================== #
st.set_page_config(layout="wide")

# ===================== PATHS ===================== #
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
base = os.path.join(BASE_DIR, "greece")

# ===================== LOAD CSV ===================== #
df = pd.read_csv("greece.csv")
image_paths = [base + p for p in df["img_path"]]

# ===================== LOAD FUNCTIONS ===================== #
def __getResults__(path):
    res_path = os.path.join(BASE_DIR, "data", path)
    with open(res_path, "rb") as f:
        data = pickle.load(f)
    return data

def __getReducer__path(path):
    redpath = os.path.join(BASE_DIR, "data", path)
    with open(redpath, "rb") as f:
        reducer = pickle.load(f)
    return reducer

def __convert__image(path):
    embedder = getEmbeddings(BASE_DIR)
    return embedder.embedder(path)

# ===================== LOAD DATA ===================== #
results = __getResults__("data.pkl")
reducer = __getReducer__path("umap_model.pkl")

# ===================== BEST K ===================== #
best_k_values = [4, 6, 7]

# ===================== TITLE ===================== #
st.title("Image Clustering Dashboard")

# ===================== BANNER ===================== #
st.markdown(
    f"""
    <div style='padding:15px;background-color:#111;border-radius:10px'>
        <h2 style='color:white'>Recommended K values: {best_k_values}</h2>
    </div>
    """,
    unsafe_allow_html=True
)

# ===================== TABS ===================== #
tab1, tab2 = st.tabs(["Clusters", "Metrics"])

# =========================================================
# ===================== TAB 1 =============================
# =========================================================
with tab1:

    # ----------- K Selection ----------- #
    k_values = [r["k"] for r in results]
    selected_k = st.selectbox("Select K", k_values, index=k_values.index(6))

    selected_result = next(r for r in results if r["k"] == selected_k)
    labels = selected_result["labels"]

    # ----------- Cluster Selection ----------- #
    clusters = np.unique(labels)
    selected_cluster = st.selectbox("Select Cluster", clusters)

    cluster_images = [
        img for img, label in zip(image_paths, labels)
        if label == selected_cluster
    ]

    st.subheader(f"Cluster {selected_cluster}")
    st.write(f"Total Images: {len(cluster_images)}")

    # ----------- Toggle View ----------- #
    show_all = st.checkbox("View all images in this cluster")

    if show_all:
        display_images = cluster_images
    else:
        display_images = cluster_images[:20]  # preview only

    # ----------- Display Grid ----------- #
    cols = st.columns(5)
    for i, img_path in enumerate(display_images):
        try:
            img = Image.open(img_path)
            cols[i % 5].image(img, use_container_width=True)
        except:
            continue

    # ===================== PREDICTION ===================== #
    st.subheader("Predict Cluster")

    uploaded_file = st.file_uploader("Upload Image", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        try:
            upload_dir = os.path.join(BASE_DIR, "uploads")
            os.makedirs(upload_dir, exist_ok=True)

            filename = str(int(time.time())) + "_" + uploaded_file.name
            save_path = os.path.join(upload_dir, filename)

            with open(save_path, "wb") as f:
                f.write(uploaded_file.getbuffer())

            img = Image.open(save_path)
            st.image(img, caption="Uploaded Image")

            # embedding
            emb = __convert__image(save_path)
            emb = np.array([emb], dtype=np.float32)

            # UMAP
            emb_umap = reducer.transform(emb).astype(np.float32)

            # predict
            kmeans = selected_result["kmeans"]
            pred = kmeans.predict(emb_umap)[0]

            st.success(f"Predicted Cluster: {pred}")

        except Exception as e:
            st.error(f"Error: {e}")

# =========================================================
# ===================== TAB 2 =============================
# =========================================================
with tab2:

    st.subheader("Clustering Metrics")

    k_list = [r["k"] for r in results]
    sil_scores = [r["silhouette"] for r in results]
    acc_scores = [r.get("accuracy", 0) for r in results]

    df_metrics = pd.DataFrame({
        "K": k_list,
        "Silhouette Score": sil_scores,
        "Accuracy": acc_scores
    })

    st.dataframe(df_metrics, use_container_width=True)
    st.line_chart(df_metrics.set_index("K"))