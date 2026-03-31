import streamlit as st
import numpy as np
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import logging

from utils.cdse import get_token_from_secrets, fetch_s1_tiff_bytes
from utils.analysis import (
    parse_coord_pair, bbox_from_latlon,
    run_analysis_from_tiff_bytes,
)
from utils.storage import append_history, load_history
from utils.geo_ui import geolocation_button, apply_qp_location

# -------------------------
# 1. PAGE CONFIGURATION
# -------------------------
st.set_page_config(
    page_title="Turkeller Surfer v3",
    page_icon="🛰️",
    layout="wide", 
    initial_sidebar_state="expanded" 
)

# -------------------------
# 2. KİMLİK DOĞRULAMA
# -------------------------
def check_password():
    APP_USER = "admin"
    APP_PASS = "altin2026"
    if "auth" not in st.session_state: st.session_state.auth = False
    if not st.session_state.auth:
        st.title("🔐 Giriş Yap")
        with st.form("login_form"):
            u = st.text_input("Kullanıcı Adı")
            p = st.text_input("Şifre", type="password")
            if st.form_submit_button("Giriş Yap", use_container_width=True):
                if u == APP_USER and p == APP_PASS:
                    st.session_state.auth = True
                    st.rerun()
                else:
                    st.error("Hatalı!")
        st.stop()  

# -------------------------
# PLOTTING FUNCTIONS 
# -------------------------
def create_2d_heatmap(Z_db_clip, X, Y, pos_mask, neg_mask, topN, cmap="jet"):
    fig = go.Figure()

    # MATLAB Stili Heatmap ve Interpolasyon
    fig.add_trace(go.Heatmap(
        z=Z_db_clip, x=X[0, :], y=Y[:, 0],
        colorscale=cmap, 
        colorbar=dict(title="VV (dB)"), name="VV",
        zsmooth="best"
    ))

    # Contour eklentileri
    if np.any(pos_mask):
        fig.add_trace(go.Contour(
            z=pos_mask.astype(int), x=X[0, :], y=Y[:, 0], showscale=False,
            contours=dict(start=0.5, end=0.5, size=1),
            line=dict(width=3, color="red"), hoverinfo="skip", name="POS",
        ))

    if np.any(neg_mask):
        fig.add_trace(go.Contour(
            z=neg_mask.astype(int), x=X[0, :], y=Y[:, 0], showscale=False,
            contours=dict(start=0.5, end=0.5, size=1),
            line=dict(width=3, color="blue"), hoverinfo="skip", name="NEG",
        ))

    for i, t in enumerate(topN, start=1):
        color = "white" if t["type"] == "POS" else "black"
        fig.add_trace(go.Scatter(
            x=[t["target_lon"]], y=[t["target_lat"]], mode="markers+text", text=[f"#{i}"],
            textposition="top center", marker=dict(symbol="cross", size=12, color=color, line=dict(width=2, color="black")),
            name=f"Top {i}"
        ))

    fig.update_layout(
        plot_bgcolor="white", paper_bgcolor="white",
        xaxis=dict(showline=True, mirror=True, linecolor='black', showgrid=True, gridcolor='lightgrey', title="Boylam"),
        yaxis=dict(showline=True, mirror=True, linecolor='black', showgrid=True, gridcolor='lightgrey', title="Enlem"),
        title=dict(text="İnteraktif Isı Haritası (MATLAB Stili)", x=0.5), height=550, margin=dict(l=50, r=50, t=50, b=50)
    )
    return fig

def create_3d_surface(Z_db_clip, X, Y, cmap="jet"):
    surf = go.Figure()
    surf.add_trace(go.Surface(
        z=Z_db_clip, x=X, y=Y, colorscale=cmap,
        contours=dict(z=dict(show=True, usecolormap=True, highlightcolor="black", project_z=True))
    ))
    surf.update_layout(
        scene=dict(
            xaxis=dict(showbackground=False, gridcolor='grey', title="Boylam"),
            yaxis=dict(showbackground=False, gridcolor='grey', title="Enlem"),
            zaxis=dict(showbackground=True, backgroundcolor="white", gridcolor='grey', title="VV (dB)")
        ),
        title=dict(text="Yumuşatılmış 3D Yüzey", x=0.5), height=550, margin=dict(l=0, r=0, t=50, b=0)
    )
    return surf

def create_matplotlib_report(Z_db, X, Y, pos_mask, neg_mask, topN, cmap="jet"):
    fig, ax = plt.subplots(figsize=(10, 8), dpi=150)
    p = ax.pcolormesh(X[0,:], Y[:,0], Z_db, shading='gouraud', cmap=cmap)
    plt.colorbar(p, ax=ax, label='VV (dB)')
    
    if np.any(pos_mask): ax.contour(X[0,:], Y[:,0], pos_mask.astype(float), levels=[0.5], colors='red', linewidths=1.5)
    if np.any(neg_mask): ax.contour(X[0,:], Y[:,0], neg_mask.astype(float), levels=[0.5], colors='black', linewidths=1.5)
        
    for i, t in enumerate(topN, start=1):
        c = 'white' if t['type'] == 'POS' else 'black'
        ax.plot(t['target_lon'], t['target_lat'], marker='x', color=c, markersize=10, mew=2)
        ax.text(t['target_lon'], t['target_lat'], f' #{i}', color=c, fontsize=12, fontweight='bold')
        
    ax.set_title("Saf MATLAB Rapor Çıktısı", fontdict={'weight':'bold', 'size':14})
    ax.set_xlabel("Boylam (X)")
    ax.set_ylabel("Enlem (Y)")
    ax.grid(color='gray', linestyle='--', linewidth=0.5, alpha=0.5)
    fig.tight_layout()
    return fig

# -------------------------
# MAIN
# -------------------------
def main():
    check_password()
    if "coord_str" not in st.session_state: st.session_state.coord_str = "40.1048440 27.7690640"
    qp_lat, qp_lon = apply_qp_location()
    if qp_lat and qp_lon: st.session_state.coord_str = f"{qp_lat:.7f} {qp_lon:.7f}"

    st.title("🛰️ Turkeller Surfer Pro v3")
    st.caption("AI Smart Interpretation | Bicubic Smoothing | MATLAB & Plotly Hybrid System")

    with st.sidebar:
        st.header("⚙️ Menü")
        with st.form("controls", clear_on_submit=False):
            scan_name = st.text_input("📝 İsim", placeholder="Kazı-1")
            coord_in = st.text_input("📌 Koordinat", value=st.session_state.coord_str)
            cap_m = st.slider("Çap (m)", 20, 300, 50)
            res_opt = st.selectbox("Çözünürlük", [120, 200, 300], index=0)
            topn = st.slider("Hedef Sayısı", 1, 15, 3)
            thr = st.slider("Anomali Eşiği", 1.5, 6.0, 2.8, 0.1)
            z_mode = st.selectbox("Algoritma", ["Lokal Z-Skoru (V3)", "Robust (Median)", "Klasik (Mean)"])
            clip_lo, clip_hi = st.slider("Kesme % Limit", 0, 99, (1, 99))
            
            st.markdown("- **Görsel Temalar**")
            theme_choice = st.selectbox("Renk Paleti", ["jet", "turbo", "parula", "coolwarm"])
            smooth_on = st.checkbox("Lee Filter (Speckle)", value=True)
            smooth_k = st.selectbox("Filtre Hassasiyeti", [3, 5, 7], index=1, disabled=(not smooth_on))

            submitted = st.form_submit_button("Analizi Başlat", type="primary", use_container_width=True)

    if submitted:
        st.session_state.coord_str = coord_in
        lat_val, lon_val = parse_coord_pair(coord_in)
        if lat_val and lon_val:
            with st.spinner("Yapay zeka modelleri ve uydu taraması devrede..."):
                try:
                    token = get_token_from_secrets()
                    bbox1 = bbox_from_latlon(lat_val, lon_val, cap_m)
                    tiff_bytes1 = fetch_s1_tiff_bytes(token, bbox1, res_opt, res_opt)
                    
                    r1 = run_analysis_from_tiff_bytes(
                        tiff_bytes1, bbox1, clip_lo, clip_hi,
                        smooth_on, int(smooth_k), z_mode, float(thr), True
                    )
                    
                    Z_db_clip = r1["Z_db_clip"]
                    X, Y = r1["X"], r1["Y"]
                    topN = r1["ranked"][: int(topn)]
                    pos_mask, neg_mask = r1["pos_mask"], r1["neg_mask"]

                    st.success("✅ Analiz Tamamlandı!")
                    
                    tab_plot, tab_mat, tab_ai = st.tabs(["🗺️ Plotly Etkileşimli", "🖼️ Saf MATLAB Statik", "🧠 Akıllı Yorumlar"])
                    
                    with tab_plot:
                        c1, c2 = st.columns(2)
                        with c1: st.plotly_chart(create_2d_heatmap(Z_db_clip, X, Y, pos_mask, neg_mask, topN, theme_choice), use_container_width=True)
                        with c2: st.plotly_chart(create_3d_surface(Z_db_clip, X, Y, theme_choice), use_container_width=True)
                        
                    with tab_mat:
                        st.pyplot(create_matplotlib_report(Z_db_clip, X, Y, pos_mask, neg_mask, topN, theme_choice))
                        
                    with tab_ai:
                        if not topN:
                            st.info("Bu parametrelerde uygun hedef bulunamadı.")
                        for i, t in enumerate(topN, start=1):
                            box_col = "🟢" if t["type"] == "POS" else "🔴"
                            with st.container(border=True):
                                st.markdown(f"### {box_col} Hedef #{i} - {t['type']} | Sınıf: **{t['depth_class']}**")
                                st.markdown(f"**🤖 AI Yorumu:** _{t['smart_comment']}_")
                                st.markdown(f"- **Skor:** `{t['score']:.2f}` | **Pik:** `{t['peak_z']:.2f}` | **Alan:** `{t['area']} px`")
                                st.markdown(f"- **Koordinat:** `{t['target_lat']:.8f}, {t['target_lon']:.8f}`")
                                
                                cA, cb = st.columns(2)
                                with cA:
                                    st.link_button("🌍 Hedefe Haritada Git", f"https://www.google.com/maps/search/?api=1&query={t['target_lat']},{t['target_lon']}", use_container_width=True)

                    append_history(
                        name=scan_name.strip() or "Tarama V3", lat=float(lat_val), lon=float(lon_val),
                        cap_m=int(cap_m), thr=float(thr), z_mode=z_mode, top=topN[:10],
                    )
                except Exception as e:
                    st.error(f"Hata: {e}")
                    logging.exception("V3 Hatası")

if __name__ == "__main__":
    main()
