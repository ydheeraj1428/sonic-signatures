
import streamlit as st
import tempfile
import os
import time
import pandas as pd

from src.matcher import SongMatcher
from src.fingerprint import fingerprint_song
from src.visualizer import (
    plot_spectrogram,
    plot_constellation
)

st.set_page_config(
    page_title="Sonic Signatures",
    page_icon="🎵",
    layout="wide"
)

# ---------------------------
# CUSTOM CSS
# ---------------------------

st.markdown("""
<style>

.main {
    padding-top: 1rem;
}

div[data-testid="metric-container"] {
    border: 1px solid rgba(255,255,255,0.15);
    padding: 15px;
    border-radius: 12px;
}

</style>
""", unsafe_allow_html=True)

matcher = SongMatcher()

# ---------------------------
# SIDEBAR
# ---------------------------

with st.sidebar:

    st.title("🎵 Sonic Signatures")

    st.markdown("---")

    if os.path.exists("songs"):

        total_songs = len([
            f for f in os.listdir("songs")
            if f.endswith(".mp3")
        ])

    else:

        total_songs = 0

    db_size = round(
        os.path.getsize(
            "database/fingerprints.pkl"
        ) / (1024 * 1024),
        2
    )

    st.metric(
        "Songs Indexed",
        total_songs
    )

    st.metric(
        "Database Size (MB)",
        db_size
    )

    st.success("System Ready")

# ---------------------------
# HEADER
# ---------------------------

st.title("🎧 Music Recognition System")
st.caption(
    "Shazam-style Audio Fingerprinting using Spectral Peak Hashing"
)

# ---------------------------
# TABS
# ---------------------------

tab1, tab2, tab3 = st.tabs(
    [
        "🎧 Identify Song",
        "📚 Library",
        "📦 Batch Mode"
    ]
)

# =====================================================
# IDENTIFY TAB
# =====================================================

with tab1:

    st.subheader("Upload Audio Clip")

    uploaded_file = st.file_uploader(
        "Supported Formats: MP3, WAV",
        type=["mp3", "wav"]
    )

    if uploaded_file:

        st.audio(uploaded_file)

        start = time.time()

        with tempfile.NamedTemporaryFile(
            delete=False,
            suffix=".wav"
        ) as tmp:

            tmp.write(
                uploaded_file.read()
            )

            temp_path = tmp.name

        with st.spinner(
            "Analyzing Audio..."
        ):

            result = fingerprint_song(
                temp_path
            )

            prediction = matcher.identify(
                temp_path
            )

        end = time.time()

        st.markdown("---")

        st.markdown(
            "## 🎯 Recognition Result"
        )

        c1, c2, c3 = st.columns(3)

        with c1:
            st.metric(
                "Matched Song",
                prediction["song"].replace(
                    ".mp3",
                    ""
                )
            )

        with c2:
            st.metric(
                "Match Score",
                prediction["score"]
            )

        with c3:
            st.metric(
                "Processing Time",
                f"{end-start:.2f}s"
            )

        st.markdown("---")

        st.markdown(
            "## 📊 Audio Analysis"
        )

        col1, col2 = st.columns(2)

        with col1:

            fig1 = plot_spectrogram(
                result["spectrogram"]
            )

            st.pyplot(
                fig1,
                use_container_width=True
            )

        with col2:

            fig2 = plot_constellation(
                result["spectrogram"],
                result["peaks"][:3000]
            )

            st.pyplot(
                fig2,
                use_container_width=True
            )

# =====================================================
# LIBRARY TAB
# =====================================================

with tab2:

    st.subheader(
        "Indexed Music Library"
    )

    if os.path.exists("songs"):

    songs = sorted([
        f for f in os.listdir("songs")
        if f.endswith(".mp3")
    ])

else:

    songs = []

    col1, col2 = st.columns(2)

    with col1:
        st.metric(
            "Total Songs",
            len(songs)
        )

    with col2:
        st.metric(
            "Database Size",
            f"{db_size} MB"
        )

    st.dataframe(
        pd.DataFrame(
            {"Song Name": songs}
        ),
        use_container_width=True
    )

# =====================================================
# BATCH TAB
# =====================================================

with tab3:

    st.subheader(
        "Batch Recognition"
    )

    uploaded_files = st.file_uploader(
        "Upload Multiple Audio Files",
        type=["mp3", "wav"],
        accept_multiple_files=True
    )

    if uploaded_files:

        results = []

        progress = st.progress(0)

        for idx, file in enumerate(
            uploaded_files
        ):

            with tempfile.NamedTemporaryFile(
                delete=False,
                suffix=".wav"
            ) as tmp:

                tmp.write(
                    file.read()
                )

                temp_path = tmp.name

            pred = matcher.identify(
                temp_path
            )

            results.append(
                {
                    "Filename":
                        file.name,
                    "Prediction":
                        pred["song"].replace(
                            ".mp3",
                            ""
                        ),
                    "Score":
                        pred["score"]
                }
            )

            progress.progress(
                (idx + 1)
                / len(uploaded_files)
            )

        df = pd.DataFrame(
            results
        )

        st.dataframe(
            df,
            use_container_width=True
        )

        csv = df.to_csv(
            index=False
        )

        st.download_button(
            "⬇ Download results.csv",
            csv,
            file_name="results.csv",
            mime="text/csv"
        )

st.markdown("---")
st.caption(
    "EE200 Mini Project • Sonic Signatures • Audio Fingerprinting System"
)

