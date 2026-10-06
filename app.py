import streamlit as st
import tempfile
import os
import time
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from collections import Counter
import io

from src.matcher import SongMatcher
from src.fingerprint import (
    fingerprint_song, load_audio, compute_spectrogram,
    find_peaks, generate_pair_hashes, generate_single_peak_hashes
)

# ─────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="Sonic Signatures",
    page_icon="🎵",
    layout="wide",
)

# ─────────────────────────────────────────────
# GLOBAL STYLES
# ─────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=SF+Pro+Display:wght@300;400;500;600;700&family=Inter:wght@300;400;500;600;700;800&display=swap');

/* ── Animated background ── */
html, body, [data-testid="stAppViewContainer"] {
    background: #060608 !important;
    color: #f0f0f5;
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
    overflow-x: hidden;
}

/* Animated gradient orbs in background */
[data-testid="stAppViewContainer"]::before {
    content: '';
    position: fixed;
    top: -30%;
    left: -20%;
    width: 60vw;
    height: 60vw;
    background: radial-gradient(circle, rgba(124,58,237,0.12) 0%, transparent 70%);
    border-radius: 50%;
    animation: orb1 12s ease-in-out infinite alternate;
    pointer-events: none;
    z-index: 0;
}
[data-testid="stAppViewContainer"]::after {
    content: '';
    position: fixed;
    bottom: -20%;
    right: -15%;
    width: 50vw;
    height: 50vw;
    background: radial-gradient(circle, rgba(56,189,248,0.10) 0%, transparent 70%);
    border-radius: 50%;
    animation: orb2 15s ease-in-out infinite alternate;
    pointer-events: none;
    z-index: 0;
}
@keyframes orb1 { 0%{transform:translate(0,0) scale(1)} 100%{transform:translate(6vw,4vw) scale(1.15)} }
@keyframes orb2 { 0%{transform:translate(0,0) scale(1)} 100%{transform:translate(-5vw,-3vw) scale(1.2)} }

[data-testid="stSidebar"] { background: rgba(10,10,18,0.85) !important; backdrop-filter: blur(20px); }

/* ── Main block ── */
[data-testid="stMainBlockContainer"] { position: relative; z-index: 1; }

/* ── Hero Card — Apple-style glassmorphism ── */
.hero {
    background: linear-gradient(135deg,
        rgba(255,255,255,0.07) 0%,
        rgba(124,58,237,0.10) 50%,
        rgba(56,189,248,0.07) 100%);
    backdrop-filter: blur(40px) saturate(180%);
    -webkit-backdrop-filter: blur(40px) saturate(180%);
    border-radius: 28px;
    padding: 2.8rem 3rem 2.2rem;
    margin-bottom: 2rem;
    border: 1px solid rgba(255,255,255,0.12);
    box-shadow:
        0 8px 32px rgba(0,0,0,0.4),
        0 1px 0 rgba(255,255,255,0.08) inset,
        0 0 80px rgba(124,58,237,0.08);
    animation: heroFadeIn 0.8s cubic-bezier(0.16,1,0.3,1) both;
    position: relative;
    overflow: hidden;
}
.hero::before {
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0;
    height: 1px;
    background: linear-gradient(90deg, transparent, rgba(255,255,255,0.25), transparent);
}
@keyframes heroFadeIn {
    from { opacity:0; transform: translateY(-20px) scale(0.98); }
    to   { opacity:1; transform: translateY(0)     scale(1); }
}
.hero h1 {
    font-size: 3rem;
    font-weight: 800;
    letter-spacing: -1.5px;
    margin: 0 0 .5rem;
    background: linear-gradient(135deg, #e0c3fc 0%, #8ec5fc 50%, #a78bfa 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    line-height: 1.1;
}
.hero p {
    color: rgba(148,163,184,0.9);
    margin: 0;
    font-size: 1rem;
    font-weight: 400;
    letter-spacing: 0.2px;
}
.hero-badge {
    display: inline-block;
    background: rgba(124,58,237,0.2);
    border: 1px solid rgba(124,58,237,0.4);
    border-radius: 20px;
    padding: 3px 12px;
    font-size: 0.72rem;
    font-weight: 600;
    color: #c4b5fd;
    letter-spacing: 0.5px;
    margin-bottom: 1rem;
    backdrop-filter: blur(10px);
}

/* ── Glass Cards (metrics) ── */
[data-testid="metric-container"] {
    background: rgba(255,255,255,0.04) !important;
    border: 1px solid rgba(255,255,255,0.09) !important;
    border-radius: 20px !important;
    padding: 1.2rem 1.4rem !important;
    backdrop-filter: blur(20px);
    -webkit-backdrop-filter: blur(20px);
    box-shadow: 0 4px 20px rgba(0,0,0,0.3), 0 1px 0 rgba(255,255,255,0.06) inset;
    transition: transform 0.25s ease, box-shadow 0.25s ease;
}
[data-testid="metric-container"]:hover {
    transform: translateY(-3px);
    box-shadow: 0 8px 30px rgba(124,58,237,0.2), 0 1px 0 rgba(255,255,255,0.1) inset;
}
@keyframes cardSlideUp {
    from { opacity:0; transform: translateY(16px); }
    to   { opacity:1; transform: translateY(0); }
}
[data-testid="stMetricLabel"]  { color: rgba(148,163,184,0.8) !important; font-size:.75rem; font-weight:500; letter-spacing:0.5px; text-transform:uppercase; }
[data-testid="stMetricValue"]  { color: #f1f5f9 !important; font-size:1.8rem !important; font-weight:700 !important; letter-spacing:-0.5px; }

/* ── Match / No-match banners ── */
.match-banner {
    background: linear-gradient(135deg, rgba(16,185,129,0.08), rgba(6,95,70,0.12));
    border: 1px solid rgba(16,185,129,0.25);
    border-radius: 20px;
    padding: 1.4rem 1.8rem;
    margin: 1rem 0;
    backdrop-filter: blur(20px);
    box-shadow: 0 4px 24px rgba(16,185,129,0.08), 0 0 40px rgba(16,185,129,0.04);
    animation: matchPop 0.5s cubic-bezier(0.16,1,0.3,1) both;
    position: relative;
    overflow: hidden;
}
.match-banner::before {
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0; height: 1px;
    background: linear-gradient(90deg, transparent, rgba(52,211,153,0.4), transparent);
}
@keyframes matchPop {
    from { opacity:0; transform: scale(0.97) translateY(8px); }
    to   { opacity:1; transform: scale(1) translateY(0); }
}
.match-banner h2 { margin:0; font-size:1.6rem; font-weight:700; color:#34d399; letter-spacing:-0.3px; }
.no-match-banner {
    background: linear-gradient(135deg, rgba(239,68,68,0.08), rgba(127,29,29,0.12));
    border: 1px solid rgba(239,68,68,0.25);
    border-radius: 20px; padding: 1.2rem 1.6rem; margin: 1rem 0;
    backdrop-filter: blur(20px);
    animation: matchPop 0.5s cubic-bezier(0.16,1,0.3,1) both;
}
.no-match-banner p { margin:0; color:#f87171; font-weight:600; font-size:1rem; }

/* ── Section labels ── */
.section-label {
    font-size: .65rem;
    font-weight: 700;
    letter-spacing: 3px;
    text-transform: uppercase;
    color: rgba(167,139,250,0.7);
    margin-bottom: .6rem;
    padding-left: 2px;
}

/* ── Tabs — pill style ── */
[data-baseweb="tab-list"] {
    background: rgba(255,255,255,0.03) !important;
    border: 1px solid rgba(255,255,255,0.08) !important;
    border-radius: 16px !important;
    padding: 4px !important;
    gap: 2px !important;
    backdrop-filter: blur(20px);
    margin-bottom: 1.5rem;
}
[data-baseweb="tab"] {
    color: rgba(100,116,139,0.9) !important;
    font-weight: 600 !important;
    font-size: .88rem !important;
    border-radius: 12px !important;
    padding: .55rem 1.3rem !important;
    transition: all 0.25s ease !important;
    border: none !important;
}
[data-baseweb="tab"]:hover {
    color: rgba(167,139,250,0.9) !important;
    background: rgba(124,58,237,0.08) !important;
}
[aria-selected="true"][data-baseweb="tab"] {
    color: #fff !important;
    background: linear-gradient(135deg, rgba(124,58,237,0.6), rgba(79,70,229,0.6)) !important;
    box-shadow: 0 2px 12px rgba(124,58,237,0.3) !important;
    border: none !important;
}

/* ── Buttons ── */
.stButton > button {
    background: linear-gradient(135deg, rgba(124,58,237,0.8), rgba(79,70,229,0.8)) !important;
    color: #fff !important;
    border: 1px solid rgba(167,139,250,0.3) !important;
    border-radius: 14px !important;
    font-weight: 600 !important;
    font-size: .9rem !important;
    padding: .65rem 1.8rem !important;
    backdrop-filter: blur(10px);
    box-shadow: 0 4px 16px rgba(124,58,237,0.25), 0 1px 0 rgba(255,255,255,0.1) inset !important;
    transition: all 0.2s cubic-bezier(0.16,1,0.3,1) !important;
    letter-spacing: 0.2px;
}
.stButton > button:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 8px 24px rgba(124,58,237,0.4), 0 1px 0 rgba(255,255,255,0.15) inset !important;
    background: linear-gradient(135deg, rgba(139,69,255,0.9), rgba(99,80,249,0.9)) !important;
}
.stButton > button:active { transform: translateY(0px) !important; }

.stDownloadButton > button {
    background: rgba(255,255,255,0.05) !important;
    color: rgba(167,139,250,0.9) !important;
    border: 1px solid rgba(167,139,250,0.25) !important;
    border-radius: 10px !important;
    font-weight: 500 !important;
    font-size: .8rem !important;
    padding: .4rem 1rem !important;
    backdrop-filter: blur(10px);
    transition: all 0.2s ease !important;
}
.stDownloadButton > button:hover {
    background: rgba(124,58,237,0.15) !important;
    border-color: rgba(167,139,250,0.5) !important;
    transform: translateY(-1px) !important;
}

/* ── File uploader ── */
[data-testid="stFileUploader"] {
    background: rgba(255,255,255,0.02) !important;
    border: 1.5px dashed rgba(100,116,139,0.3) !important;
    border-radius: 20px !important;
    padding: 1.5rem !important;
    transition: all 0.3s ease;
    backdrop-filter: blur(10px);
}
[data-testid="stFileUploader"]:hover {
    border-color: rgba(124,58,237,0.5) !important;
    background: rgba(124,58,237,0.03) !important;
}

/* ── Dataframe ── */
[data-testid="stDataFrame"] {
    border-radius: 16px !important;
    overflow: hidden;
    border: 1px solid rgba(255,255,255,0.07) !important;
    backdrop-filter: blur(10px);
}

/* ── Selectbox ── */
[data-baseweb="select"] > div {
    background: rgba(255,255,255,0.04) !important;
    border: 1px solid rgba(255,255,255,0.1) !important;
    border-radius: 14px !important;
    backdrop-filter: blur(10px);
    transition: all 0.2s ease;
}
[data-baseweb="select"] > div:hover {
    border-color: rgba(124,58,237,0.4) !important;
}

/* ── Audio player ── */
audio {
    width: 100%;
    border-radius: 14px;
    filter: invert(0.85) hue-rotate(220deg) saturate(1.5);
}

/* ── Progress bar ── */
[data-testid="stProgressBar"] > div {
    background: linear-gradient(90deg, #7c3aed, #60a5fa) !important;
    border-radius: 4px;
    box-shadow: 0 0 12px rgba(124,58,237,0.5);
}

/* ── Divider ── */
hr {
    border: none !important;
    height: 1px !important;
    background: linear-gradient(90deg, transparent, rgba(255,255,255,0.08), transparent) !important;
    margin: 1.5rem 0 !important;
}

/* ── Spinner ── */
[data-testid="stSpinner"] { color: #a78bfa !important; }

/* ── Info/success/warning boxes ── */
[data-testid="stAlert"] {
    border-radius: 16px !important;
    backdrop-filter: blur(20px) !important;
    border: 1px solid rgba(255,255,255,0.08) !important;
}

/* Scroll reveal removed — caused shaking on Windows due to Streamlit reruns */

/* ── Pyplot charts glass frame ── */
[data-testid="stImage"], .stPlotlyChart {
    border-radius: 20px;
    overflow: hidden;
    box-shadow: 0 4px 24px rgba(0,0,0,0.4);
    border: 1px solid rgba(255,255,255,0.07);
    transition: box-shadow 0.3s ease;
}
[data-testid="stImage"]:hover {
    box-shadow: 0 8px 32px rgba(124,58,237,0.2);
}

/* ── Hide pyplot fullscreen/expand button ── */
[data-testid="stElementToolbar"] {
    display: none !important;
}
button[title="View fullscreen"],
button[aria-label="View fullscreen"],
button[title="Fullscreen"],
[data-testid="StyledFullScreenButton"] {
    display: none !important;
    visibility: hidden !important;
    pointer-events: none !important;
}
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# HERO
# ─────────────────────────────────────────────
st.markdown("""
<div class="hero">
    <div class="hero-badge">✦ EE200 Course Project</div>
    <h1>🎵 Sonic Signatures</h1>
    <p>Audio fingerprinting &nbsp;·&nbsp; Mini Shazam &nbsp;·&nbsp; Built with spectrograms &amp; hash matching</p>
</div>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# LOAD MATCHERS
# ─────────────────────────────────────────────
@st.cache_resource
def load_matcher():
    return SongMatcher(mode="pair")

@st.cache_resource
def load_matcher_single():
    return SongMatcher(mode="single")

matcher = load_matcher()

# ─────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────
def fig_to_bytes(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=150, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    buf.seek(0)
    return buf.getvalue()

def dl_button(fig, filename, label="⬇️ Download"):
    st.download_button(label, fig_to_bytes(fig), file_name=filename, mime="image/png", key=filename+str(id(fig)))

def _dark_fig(figsize=(10, 4)):
    fig, ax = plt.subplots(figsize=figsize)
    fig.patch.set_facecolor("#12121c")
    ax.set_facecolor("#1a1a2e")
    for spine in ax.spines.values():
        spine.set_edgecolor("#334155")
    ax.tick_params(colors="#94a3b8", labelsize=9)
    ax.xaxis.label.set_color("#94a3b8")
    ax.yaxis.label.set_color("#94a3b8")
    ax.title.set_color("#e2e8f0")
    return fig, ax

def plot_offset_histogram_fixed(offsets, title="Offset Vote Distribution"):
    counts = Counter(offsets)
    if not counts:
        fig, ax = _dark_fig()
        ax.set_title(title, fontsize=13, fontweight="bold", pad=12)
        ax.text(0.5, 0.5, "No offset data", ha="center", va="center",
                transform=ax.transAxes, color="#64748b", fontsize=11)
        return fig
    top_n  = sorted(counts.most_common(20), key=lambda x: x[0])
    labels = [str(x[0]) for x in top_n]
    votes  = [x[1]      for x in top_n]
    max_v  = max(votes)
    norm_v = np.array(votes) / max_v
    cmap   = plt.colormaps["plasma"]
    colors = [cmap(0.35 + 0.6 * v) for v in norm_v]
    fig, ax = _dark_fig(figsize=(10, 4))
    ax.bar(labels, votes, color=colors, edgecolor="#0d0d14", linewidth=0.6, zorder=3)
    best_idx = int(np.argmax(votes))
    ax.bar([labels[best_idx]], [votes[best_idx]], color="#a78bfa",
           edgecolor="#c4b5fd", linewidth=1.2, zorder=4)
    ax.annotate(f"best: {labels[best_idx]}",
                xy=(best_idx, votes[best_idx]),
                xytext=(best_idx, votes[best_idx] + max_v * 0.04),
                color="#c4b5fd", fontsize=8.5, ha="center")
    ax.set_title(title, fontsize=13, fontweight="bold", pad=12)
    ax.set_xlabel("Time Offset (frames)", fontsize=9)
    ax.set_ylabel("Hash Votes", fontsize=9)
    ax.grid(axis="y", alpha=0.15, color="#ffffff", zorder=0)
    ax.set_axisbelow(True)
    if len(labels) > 12:
        plt.xticks(rotation=45, ha="right", fontsize=7.5)
    fig.tight_layout()
    return fig

def _styled_spectrogram(spec):
    import librosa.display
    fig, ax = plt.subplots(figsize=(10, 4))
    fig.patch.set_facecolor("#12121c")
    ax.set_facecolor("#1a1a2e")
    img = librosa.display.specshow(spec, x_axis="time", y_axis="hz", ax=ax, cmap="magma")
    fig.colorbar(img, ax=ax, format="%+2.0f dB").ax.tick_params(colors="#94a3b8")
    ax.set_title("Spectrogram", fontsize=13, fontweight="bold", color="#e2e8f0", pad=10)
    ax.tick_params(colors="#94a3b8")
    for spine in ax.spines.values(): spine.set_edgecolor("#334155")
    ax.xaxis.label.set_color("#94a3b8")
    ax.yaxis.label.set_color("#94a3b8")
    fig.tight_layout()
    return fig

def _styled_constellation(peaks, title="Constellation Map"):
    fig, ax = _dark_fig(figsize=(10, 4))
    ax.set_facecolor("#0d0d18")
    if peaks:
        time_idx = [p[0] for p in peaks]
        freq_idx = [p[1] for p in peaks]
        ax.scatter(time_idx, freq_idx, s=4, alpha=0.6, c="#a78bfa", linewidths=0)
    ax.set_title(title, fontsize=13, fontweight="bold", pad=10)
    ax.set_xlabel("Time Frame", fontsize=9)
    ax.set_ylabel("Frequency Bin", fontsize=9)
    ax.grid(alpha=0.08, color="#ffffff")
    fig.tight_layout()
    return fig

def run_identification(uploaded_file):
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
        tmp.write(uploaded_file.read())
        temp_path = tmp.name
    try:
        t0 = time.time()
        with st.spinner("⚙️ Generating fingerprint …"):
            result = fingerprint_song(temp_path)
        prediction = matcher.identify_fingerprint(result)
        elapsed = round(time.time() - t0, 2)
        return result, prediction, elapsed
    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)

# ─────────────────────────────────────────────
# TABS
# ─────────────────────────────────────────────
# Tab switching via query params
_params = st.query_params
_active_tab = int(_params.get("tab", 0))
if st.session_state.get("goto_exp"):
    st.session_state["goto_exp"] = False
    st.query_params["tab"] = "3"
    st.rerun()

TAB_NAMES = ["📚  Library", "🎧  Identify", "📦  Batch", "🔬  Experiments"]
tab_lib, tab_id, tab_batch, tab_exp = st.tabs(TAB_NAMES)

# Highlight active tab via JS
st.markdown(f"""
<script>
(function() {{
    function clickTab() {{
        const tabs = window.parent.document.querySelectorAll('[data-baseweb="tab"]');
        const idx = {_active_tab};
        if (tabs && tabs[idx] && !tabs[idx].getAttribute('aria-selected') === 'true') {{
            tabs[idx].click();
        }}
    }}
    setTimeout(clickTab, 400);
    setTimeout(clickTab, 800);
}})();
</script>
""", unsafe_allow_html=True)

# ══════════════════════════════════════════════
# TAB 1 — LIBRARY
# ══════════════════════════════════════════════
with tab_lib:
    st.markdown('<p class="section-label">Song Database</p>', unsafe_allow_html=True)
    songs = sorted([f for f in os.listdir("songs") if f.endswith(".mp3")])             if os.path.exists("songs") else []

    c1, c2, c3 = st.columns(3)
    c1.metric("Indexed Songs", len(songs) if songs else "N/A")
    c2.metric("Database", "Ready ✓")
    c3.metric("Fingerprint Mode", "Pair Hash")
    st.markdown("---")

    if songs:
        st.markdown('<p class="section-label">Browse & Play</p>', unsafe_allow_html=True)

        if "playing_song" not in st.session_state:
            st.session_state["playing_song"] = None
        if "goto_exp" not in st.session_state:
            st.session_state["goto_exp"] = False

        cols_per_row = 3
        for row_start in range(0, len(songs), cols_per_row):
            row_songs = songs[row_start:row_start+cols_per_row]
            cols = st.columns(cols_per_row)
            for col, song in zip(cols, row_songs):
                song_name = song.replace(".mp3","")
                is_playing = st.session_state["playing_song"] == song
                with col:
                    st.markdown(f"""
                    <div style="
                        background: {"rgba(124,58,237,0.12)" if is_playing else "rgba(255,255,255,0.04)"};
                        border: 1px solid {"rgba(167,139,250,0.4)" if is_playing else "rgba(255,255,255,0.09)"};
                        border-radius: 16px;
                        padding: 1rem 1.2rem 0.8rem;
                        margin-bottom: 0.4rem;
                        backdrop-filter: blur(10px);
                    ">
                        <div style="font-size:1.4rem; margin-bottom:0.3rem;">{"▶️" if is_playing else "🎵"}</div>
                        <div style="font-weight:600; font-size:0.82rem; color:#e2e8f0;
                                    white-space:nowrap; overflow:hidden; text-overflow:ellipsis;
                                    margin-bottom:0.5rem;" title="{song_name}">
                            {song_name}
                        </div>
                    </div>
                    """, unsafe_allow_html=True)

                    # Audio player inline — only for playing song
                    if is_playing:
                        audio_path = os.path.join("songs", song)
                        if os.path.exists(audio_path):
                            with open(audio_path, "rb") as f_:
                                st.audio(f_.read(), format="audio/mp3")
                        if st.button("⏹ Stop", key=f"stop_{song}"):
                            st.session_state["playing_song"] = None
                            st.rerun()
                    else:
                        btn_col1, btn_col2 = st.columns(2)
                        if btn_col1.button("▶ Play", key=f"play_{song}"):
                            st.session_state["playing_song"] = song
                            st.rerun()
                        if btn_col2.button("🔬 Exp", key=f"use_exp_{song}"):
                            audio_path = os.path.join("songs", song)
                            if os.path.exists(audio_path):
                                with open(audio_path, "rb") as f_:
                                    ab = f_.read()
                                st.session_state["exp_song_bytes"] = ab
                                st.session_state["exp_song_name"]  = song_name
                                for k in ["exp_key","exp_done","noise_data",
                                          "pitch_data","exp_spec","exp_peaks"]:
                                    st.session_state.pop(k, None)
                                st.session_state["exp_loaded_msg"] = song_name
                                st.rerun()
                    if st.session_state.get("exp_loaded_msg") == song_name:
                        st.markdown("""<div style="background:rgba(124,58,237,0.2);
                            border:1.5px solid #a78bfa;border-radius:10px;
                            padding:6px 8px;font-size:0.78rem;color:#e2e8f0;
                            text-align:center;margin-top:2px;">
                            ✅ Go to 🔬 <b>Experiments</b> tab
                            </div>""", unsafe_allow_html=True)

    else:
        st.info("Songs folder not available on this deployment.")

# ══════════════════════════════════════════════
# TAB 2 — IDENTIFY
# ══════════════════════════════════════════════
with tab_id:
    st.markdown('<p class="section-label">Upload a clip to identify</p>', unsafe_allow_html=True)
    uploaded_file = st.file_uploader("Drop an audio clip here (MP3 or WAV)",
                                     type=["mp3", "wav"], key="single_upload")
    if uploaded_file:
        st.audio(uploaded_file)
        if st.button("🔍 Identify Song", key="btn_identify"):
            st.session_state["id_result"] = run_identification(uploaded_file)

        if "id_result" not in st.session_state:
            st.info("Click **Identify Song** button to start.")
        else:
            result, prediction, elapsed = st.session_state["id_result"]
            st.markdown("---")
            if prediction["song"]:
                song_clean = prediction["song"].replace(".mp3", "")
                st.markdown(f'<div class="match-banner"><h2>🎵 &nbsp; {song_clean}</h2></div>',
                            unsafe_allow_html=True)
                m1, m2, m3, m4 = st.columns(4)
                raw_score = prediction["score"]
                if raw_score >= 500: conf_display = 99
                elif raw_score >= 200: conf_display = 90 + int((raw_score-200)/30)
                elif raw_score >= 100: conf_display = 75 + int((raw_score-100)/10)
                elif raw_score >= 50:  conf_display = 50 + int((raw_score-50)*0.5)
                else: conf_display = max(10, int(raw_score*0.8))
                conf_display = min(99, conf_display)
                m1.metric("Match Score", raw_score)
                m2.metric("Confidence", f"{conf_display}%")
                best_off = prediction["top_offsets"][0][0] if prediction["top_offsets"] else "—"
                m3.metric("Best Offset", best_off)
                m4.metric("Time Taken", f"{elapsed}s")
            else:
                st.markdown(f'<div class="no-match-banner"><p>❌ &nbsp; No confident match '
                            f'(status: {prediction["status"]})</p></div>', unsafe_allow_html=True)
            st.markdown("---")
            st.markdown('<p class="section-label">Signal Analysis</p>', unsafe_allow_html=True)
            col1, col2 = st.columns(2)
            with col1:
                st.markdown("**Spectrogram**")
                fig1 = _styled_spectrogram(result["spectrogram"])
                st.pyplot(fig1, use_container_width=True)
                dl_button(fig1, "spectrogram.png")
                plt.close(fig1)
            with col2:
                st.markdown("**Constellation Map**")
                fig2 = _styled_constellation(result["peaks"][:1000])
                st.pyplot(fig2, use_container_width=True)
                dl_button(fig2, "constellation.png")
                plt.close(fig2)
            if prediction["song"] and prediction["offsets"]:
                st.markdown("---")
                st.markdown('<p class="section-label">Hash Matching</p>', unsafe_allow_html=True)
                col3, col4 = st.columns([3, 1])
                with col3:
                    fig3 = plot_offset_histogram_fixed(prediction["offsets"])
                    st.pyplot(fig3, use_container_width=True)
                    dl_button(fig3, "offset_histogram.png")
                    plt.close(fig3)
                with col4:
                    st.markdown("**Top Offsets**")
                    df_off = pd.DataFrame(prediction["top_offsets"], columns=["Offset", "Votes"])
                    st.dataframe(df_off, use_container_width=True, hide_index=True)
                    st.markdown("<small style='color:#64748b'>Tall spike = confident match.</small>",
                                unsafe_allow_html=True)

# ══════════════════════════════════════════════
# TAB 3 — BATCH
# ══════════════════════════════════════════════
with tab_batch:
    st.markdown('<p class="section-label">Batch Recognition</p>', unsafe_allow_html=True)
    st.caption("Upload multiple clips — results exported as `results.csv`.")
    uploaded_files = st.file_uploader("Drop audio clips here", type=["mp3", "wav"],
                                      accept_multiple_files=True, key="batch_upload")
    if uploaded_files:
        if st.button("🚀 Run Batch Recognition", key="btn_batch"):
            results = []
            st.toast("⚙️ Batch recognition started!", icon="🎵")
            progress_bar = st.progress(0, text="⚙️ Starting…")
            status_box = st.empty()
            for idx, file in enumerate(uploaded_files):
                status_box.markdown(
                    f"<small style='color:#94a3b8'>🔍 Processing <b>{file.name}</b> "
                    f"({idx+1}/{len(uploaded_files)})…</small>",
                    unsafe_allow_html=True)
                with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
                    tmp.write(file.read())
                    temp_path = tmp.name
                try:
                    fp_result = fingerprint_song(temp_path)
                    pred = matcher.identify_fingerprint(fp_result)
                    label = pred["song"].replace(".mp3", "") if pred["song"] else "Unknown"
                    raw_score = pred.get("score", 0)
                    if raw_score >= 500: conf = 99
                    elif raw_score >= 200: conf = 90 + int((raw_score-200)/30)
                    elif raw_score >= 100: conf = 75 + int((raw_score-100)/10)
                    elif raw_score >= 50:  conf = 50 + int((raw_score-50)*0.5)
                    else: conf = max(10, int(raw_score*0.8))
                    conf = min(99, conf)
                    best_off = pred["top_offsets"][0][0] if pred.get("top_offsets") else "—"
                    results.append({
                        "filename":   file.name,
                        "prediction": label,
                        "score":      raw_score,
                        "confidence": f"{conf}%" if label != "Unknown" else "—",
                        "best_offset": best_off,
                        "status":     pred.get("status", "—"),
                    })
                finally:
                    if os.path.exists(temp_path):
                        os.remove(temp_path)
                progress_bar.progress((idx+1)/len(uploaded_files),
                                      text=f"✅ Done {idx+1}/{len(uploaded_files)}")
            status_box.empty()
            progress_bar.empty()
            st.session_state["batch_df"] = pd.DataFrame(results)

        if "batch_df" in st.session_state:
            df = st.session_state["batch_df"]
            n_matched = (df["prediction"] != "Unknown").sum()
            mc1, mc2, mc3 = st.columns(3)
            mc1.metric("Total Clips", len(df))
            mc2.metric("Matched", n_matched)
            mc3.metric("Unrecognised", len(df)-n_matched)
            st.markdown("---")
            st.dataframe(df, use_container_width=True, hide_index=True)
            csv = df.to_csv(index=False)
            st.download_button("⬇️  Download results.csv", csv,
                               file_name="results.csv", mime="text/csv")
        else:
            st.info("Upload clips and click **Run Batch Recognition** to start.")

# ══════════════════════════════════════════════
# TAB 4 — EXPERIMENTS (Q3A)
# ══════════════════════════════════════════════
with tab_exp:
    st.markdown('<p class="section-label">Q3A Experiments</p>', unsafe_allow_html=True)
    st.caption("Upload **any** audio clip or pick one from Library → 🔬 Use for Experiment.")

    import librosa
    import librosa.display

    # ── Audio source ─────────────────────────────────────────────
    src_col1, src_col2 = st.columns([2,1])
    with src_col1:
        exp_file = st.file_uploader("Upload any clip (MP3 / WAV)",
                                    type=["mp3","wav"], key="exp_upload")
    with src_col2:
        st.markdown("<br>", unsafe_allow_html=True)
        if "exp_song_name" in st.session_state:
            st.success(f"🎵 From Library: **{st.session_state['exp_song_name']}**")
            if st.button("❌ Clear library song", key="clear_lib_song"):
                for k in ["exp_song_bytes","exp_song_name","exp_key","exp_audio",
                          "exp_sr","exp_done","noise_data","pitch_data"]:
                    st.session_state.pop(k, None)
                st.rerun()

    # Load audio — from uploaded file or library
    audio = None; sr = None
    if exp_file:
        exp_key = "upload_" + exp_file.name + str(exp_file.size)
        if st.session_state.get("exp_key") != exp_key:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
                tmp.write(exp_file.read())
                exp_path = tmp.name
            try:
                _audio, _sr = load_audio(exp_path)
            finally:
                if os.path.exists(exp_path): os.remove(exp_path)
            st.session_state["exp_audio"] = _audio
            st.session_state["exp_sr"]    = _sr
            st.session_state["exp_key"]   = exp_key
            # Clear ALL experiment results so new song starts fresh
            for k in ["noise_data","pitch_data","exp_done","exp_spec","exp_peaks",
                      "exp_selector"]:
                st.session_state.pop(k, None)
            st.rerun()  # force a clean render with no stale graphs
        audio = st.session_state["exp_audio"]
        sr    = st.session_state["exp_sr"]

    elif "exp_song_bytes" in st.session_state:
        exp_key = "lib_" + st.session_state.get("exp_song_name","")
        if st.session_state.get("exp_key") != exp_key:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as tmp:
                tmp.write(st.session_state["exp_song_bytes"])
                exp_path = tmp.name
            try:
                _audio, _sr = load_audio(exp_path)
            finally:
                if os.path.exists(exp_path): os.remove(exp_path)
            st.session_state["exp_audio"] = _audio
            st.session_state["exp_sr"]    = _sr
            st.session_state["exp_key"]   = exp_key
            # Clear ALL experiment results so new song starts fresh
            for k in ["noise_data","pitch_data","exp_done","exp_spec","exp_peaks",
                      "exp_selector"]:
                st.session_state.pop(k, None)
            st.rerun()  # force a clean render with no stale graphs
        audio = st.session_state["exp_audio"]
        sr    = st.session_state["exp_sr"]

    if audio is None:
        st.markdown("""
        <div style="background:rgba(124,58,237,0.10);border:1.5px solid rgba(167,139,250,0.4);
        border-radius:16px;padding:1.2rem 1.5rem;margin-bottom:1rem;">
        <div style="font-size:1.1rem;font-weight:700;color:#a78bfa;margin-bottom:0.4rem;">
        How to use Experiments</div>
        <div style="color:#cbd5e1;font-size:0.92rem;line-height:1.7;">
        <b>Option 1:</b> Upload any MP3/WAV clip using the uploader above.<br>
        <b>Option 2:</b> Go to <b>📚 Library</b> tab → find any song → click <b>🔬 Exp</b> button → 
        then come back here to this <b>🔬 Experiments</b> tab.
        </div>
        </div>
        """, unsafe_allow_html=True)
    else:
        clip_duration = len(audio) / sr
        st.success(f"✅ Ready: **{clip_duration:.2f}s** clip at {sr} Hz")
        if clip_duration < 10:
            st.warning("⚠️ Short clip (<10s) — noise & pitch experiments may be unreliable.")

        st.markdown("---")

        # ── Experiment picker ─────────────────────────────────────
        EXP_OPTIONS = [
            "📊 Exp 1 — Full Song DFT",
            "🪟 Exp 2 — Short vs Long Window Spectrogram",
            "✨ Exp 3 — Constellation Map",
            "⚖️ Exp 4 — Single Peak vs Pair Hash",
            "🔊 Exp 5 — Noise Robustness",
            "🎼 Exp 6 — Pitch Shift Robustness",
            "⚡ Run All Experiments",
        ]
        sel = st.selectbox("Select experiment", EXP_OPTIONS, key="exp_selector")
        run_btn = st.button("▶ Run", key="run_exp_btn", type="primary")
        st.markdown("---")

        # ── Helper functions ──────────────────────────────────────
        def do_exp1(audio, sr):
            st.subheader("📊 Exp 1 — Full Song DFT")
            st.caption("Why a single DFT isn't enough — no timing info.")
            prog = st.progress(0, text="Computing FFT…")
            fft_mag = np.abs(np.fft.rfft(audio))
            freqs   = np.fft.rfftfreq(len(audio), d=1/sr)
            prog.progress(50, text="Plotting…")
            fig, ax = _dark_fig(figsize=(10, 3))
            ax.plot(freqs, fft_mag, color="#a78bfa", linewidth=0.6, alpha=0.85)
            ax.set_xlabel("Frequency (Hz)"); ax.set_ylabel("Magnitude")
            ax.set_title("DFT Magnitude — Full Song", fontsize=13, fontweight="bold")
            ax.grid(alpha=0.12); fig.tight_layout()
            prog.empty()
            st.pyplot(fig, use_container_width=True)
            dl_button(fig, "exp1_dft.png"); plt.close(fig)
            st.markdown("**Observation:** All timing info is lost — you see *which* frequencies exist but not *when* each note was played. This is why we need a spectrogram.")

        def do_exp2(audio, sr):
            st.subheader("🪟 Exp 2 — Short vs Long Window Spectrogram")
            col_a, col_b = st.columns(2)
            for col, n_fft, label, note, fname in [
                (col_a, 512, "Short Window (n_fft=512)",
                 "Good **time** resolution, poor **frequency** resolution.", "exp2_short.png"),
                (col_b, 4096, "Long Window (n_fft=4096)",
                 "Good **frequency** resolution, poor **time** resolution.", "exp2_long.png"),
            ]:
                prog = col.progress(0, text=f"Computing {label}…")
                spec = compute_spectrogram(audio, n_fft=n_fft)
                f2, a2 = plt.subplots(figsize=(6,3))
                f2.patch.set_facecolor("#12121c"); a2.set_facecolor("#1a1a2e")
                img = librosa.display.specshow(spec, x_axis="time", y_axis="hz",
                                               ax=a2, cmap="magma", sr=sr, hop_length=512)
                f2.colorbar(img, ax=a2, format="%+2.0f dB").ax.tick_params(colors="#94a3b8")
                a2.set_title(label, fontsize=11, fontweight="bold", color="#e2e8f0")
                a2.tick_params(colors="#94a3b8")
                for sp in a2.spines.values(): sp.set_edgecolor("#334155")
                a2.xaxis.label.set_color("#94a3b8"); a2.yaxis.label.set_color("#94a3b8")
                f2.tight_layout()
                prog.empty()
                with col:
                    st.pyplot(f2, use_container_width=True)
                    dl_button(f2, fname); plt.close(f2)
                    st.caption(note)
            st.markdown("**Conclusion:** Default n_fft=2048 balances both resolutions for fingerprinting.")

        def do_exp3(audio, sr):
            st.subheader("✨ Exp 3 — Constellation Map")
            prog = st.progress(0, text="Computing spectrogram…")
            spec_d = compute_spectrogram(audio)
            prog.progress(50, text="Finding peaks…")
            peaks_a = find_peaks(spec_d)
            prog.empty()
            col1, col2 = st.columns(2)
            with col1:
                f3 = _styled_spectrogram(spec_d)
                st.pyplot(f3, use_container_width=True)
                dl_button(f3, "exp3_spec.png"); plt.close(f3)
                st.caption("Full spectrogram")
            with col2:
                f4 = _styled_constellation(peaks_a[:2000], title=f"Constellation ({len(peaks_a)} peaks)")
                st.pyplot(f4, use_container_width=True)
                dl_button(f4, "exp3_const.png"); plt.close(f4)
                st.caption(f"{len(peaks_a)} peaks found")
            st.markdown(f"**Observation:** From dense spectrogram → only **{len(peaks_a)} peaks** kept. Sparse but robust.")
            st.session_state["exp_spec"]  = spec_d
            st.session_state["exp_peaks"] = peaks_a
            return spec_d, peaks_a

        def do_exp4(audio, sr):
            st.subheader("⚖️ Exp 4 — Single Peak vs Pair Hash")
            prog = st.progress(0, text="Computing fingerprints…")
            spec_d  = st.session_state.get("exp_spec",  compute_spectrogram(audio))
            peaks_a = st.session_state.get("exp_peaks", find_peaks(spec_d))
            prog.progress(40, text="Running pair matcher…")
            pr = load_matcher().identify_fingerprint({
                "hashes": generate_pair_hashes(peaks_a),
                "spectrogram": spec_d, "peaks": peaks_a, "mode": "pair"})
            prog.progress(70, text="Running single matcher…")
            sr2 = load_matcher_single().identify_fingerprint({
                "hashes": generate_single_peak_hashes(peaks_a),
                "spectrogram": spec_d, "peaks": peaks_a, "mode": "single"})
            prog.empty()
            mc1, mc2 = st.columns(2)
            with mc1:
                st.markdown("**Pair Hash**")
                st.metric("Song",  pr.get("song","—") or "No match")
                st.metric("Score", pr["score"])
            with mc2:
                st.markdown("**Single Peak**")
                st.metric("Song",  sr2.get("song","—") or "No match")
                st.metric("Score", sr2["score"])
            f5, a5 = _dark_fig(figsize=(6,4))
            mx = max(sr2["score"], pr["score"], 1)
            bars = a5.bar(["Single Peak","Pair Hash"], [sr2["score"], pr["score"]],
                          color=["#60a5fa","#a78bfa"], edgecolor="#0d0d14", width=0.4)
            for bar, val in zip(bars, [sr2["score"], pr["score"]]):
                a5.text(bar.get_x()+bar.get_width()/2, bar.get_height()+mx*0.02,
                        str(val), ha="center", color="#e2e8f0", fontsize=11, fontweight="bold")
            a5.set_title("Match Score Comparison", fontsize=13, fontweight="bold")
            a5.set_ylabel("Match Score"); a5.grid(axis="y", alpha=0.15); f5.tight_layout()
            st.pyplot(f5, use_container_width=True)
            dl_button(f5, "exp4_compare.png"); plt.close(f5)
            st.markdown("**Why pairs win:** Hash encodes (f1, f2, Δt) — all three must match. Random collision drops exponentially vs single-peak.")

        def do_exp5(audio, sr):
            st.subheader("🔊 Exp 5 — Noise Robustness")
            noise_levels = [0.0, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2]
            scores = []
            prog = st.progress(0, text="Starting noise experiment…")
            for i, std in enumerate(noise_levels):
                prog.progress(int((i+1)/len(noise_levels)*100),
                              text=f"Noise std={std} → testing… ({i+1}/{len(noise_levels)})")
                noisy   = audio + np.random.normal(0, std, len(audio))
                spec_n  = compute_spectrogram(noisy)
                peaks_n = find_peaks(spec_n)
                pred_n  = load_matcher().identify_fingerprint(
                    {"hashes": generate_pair_hashes(peaks_n)}, min_score=1)
                scores.append(pred_n["score"])
            prog.empty()
            st.session_state["noise_data"] = (noise_levels, scores)
            f6, a6 = _dark_fig(figsize=(9,4))
            a6.plot([str(n) for n in noise_levels], scores, marker="o", color="#a78bfa",
                    linewidth=2, markersize=7, markerfacecolor="#60a5fa", markeredgecolor="#0d0d14")
            a6.fill_between(range(len(noise_levels)), scores, alpha=0.15, color="#a78bfa")
            a6.set_title("Recognition Score vs Noise Level", fontsize=13, fontweight="bold")
            a6.set_xlabel("Noise Std"); a6.set_ylabel("Match Score")
            a6.grid(alpha=0.15); f6.tight_layout()
            st.pyplot(f6, use_container_width=True)
            dl_button(f6, "exp5_noise.png"); plt.close(f6)
            base = scores[0] if scores[0] > 0 else 1
            drop = next((i for i,s in enumerate(scores) if s < base*0.5), None)
            if drop:
                st.markdown(f"**Result:** Held until std={noise_levels[drop-1]}, dropped at std={noise_levels[drop]} ({scores[drop]} vs {scores[0]} baseline). Peaks get masked beyond this threshold.")
            else:
                st.markdown(f"**Result:** Robust across all noise levels (baseline={scores[0]}).")

        def do_exp6(audio, sr):
            st.subheader("🎼 Exp 6 — Pitch Shift Robustness")
            semis = [0, 1, 2, 3, -1, -2]
            scores = []
            prog = st.progress(0, text="Starting pitch-shift experiment…")
            for i, semi in enumerate(semis):
                prog.progress(int((i+1)/len(semis)*100),
                              text=f"Pitch {semi:+d} semitones… ({i+1}/{len(semis)})")
                shifted = librosa.effects.pitch_shift(audio, sr=sr, n_steps=semi)
                spec_p  = compute_spectrogram(shifted)
                peaks_p = find_peaks(spec_p)
                pred_p  = load_matcher().identify_fingerprint(
                    {"hashes": generate_pair_hashes(peaks_p)}, min_score=1)
                scores.append(pred_p["score"])
            prog.empty()
            st.session_state["pitch_data"] = (semis, scores)
            s0 = scores[semis.index(0)] if 0 in semis else 1
            colors = ["#34d399" if s==0 else "#fbbf24" if scores[i]>s0*0.5 else "#f87171"
                      for i,s in enumerate(semis)]
            f7, a7 = _dark_fig(figsize=(9,4))
            a7.bar([str(s) for s in semis], scores, color=colors, edgecolor="#0d0d14", linewidth=0.6)
            a7.set_title("Match Score vs Pitch Shift", fontsize=13, fontweight="bold")
            a7.set_xlabel("Pitch Shift (semitones)"); a7.set_ylabel("Match Score")
            a7.grid(axis="y", alpha=0.15); f7.tight_layout()
            st.pyplot(f7, use_container_width=True)
            dl_button(f7, "exp6_pitch.png"); plt.close(f7)
            failed = [s for s,sc in zip(semis,scores) if s!=0 and sc<=s0*0.5]
            if failed:
                st.markdown(f"**Result:** Failed at {failed} semitone(s). Dropped from {s0} → {min(scores[semis.index(s)] for s in failed)}. **Why:** Absolute frequency bins shift — no hashes match. **Fix:** Use frequency ratios f2/f1.")
            else:
                st.markdown(f"**Result:** Robust at all shifts (baseline={s0}). Try larger clips for clearer degradation.")

        # ── Run logic ─────────────────────────────────────────────
        if run_btn:
            idx = EXP_OPTIONS.index(sel)
            st.session_state["exp_done"] = idx
            if   idx == 0: do_exp1(audio, sr)
            elif idx == 1: do_exp2(audio, sr)
            elif idx == 2: do_exp3(audio, sr)
            elif idx == 3: do_exp4(audio, sr)
            elif idx == 4: do_exp5(audio, sr)
            elif idx == 5: do_exp6(audio, sr)
            elif idx == 6:
                do_exp1(audio, sr); st.markdown("---")
                do_exp2(audio, sr); st.markdown("---")
                do_exp3(audio, sr); st.markdown("---")
                do_exp4(audio, sr); st.markdown("---")
                do_exp5(audio, sr); st.markdown("---")
                do_exp6(audio, sr)

        elif "exp_done" in st.session_state:
            idx = st.session_state["exp_done"]
            if   idx == 0: do_exp1(audio, sr)
            elif idx == 1: do_exp2(audio, sr)
            elif idx == 2: do_exp3(audio, sr)
            elif idx == 3: do_exp4(audio, sr)
            elif idx == 4: do_exp5(audio, sr)
            elif idx == 5: do_exp6(audio, sr)
            elif idx == 6:
                do_exp1(audio, sr); st.markdown("---")
                do_exp2(audio, sr); st.markdown("---")
                do_exp3(audio, sr); st.markdown("---")
                do_exp4(audio, sr); st.markdown("---")
                do_exp5(audio, sr); st.markdown("---")
                do_exp6(audio, sr)
        else:
            st.info("👆 Select an experiment and click **▶ Run**")
