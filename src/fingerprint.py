
import librosa
import numpy as np
from scipy.ndimage import maximum_filter


def load_audio(file_path, sr=22050):
    """
    Load audio and convert to mono.
    """
    audio, sr = librosa.load(file_path, sr=sr, mono=True)
    return audio, sr


def compute_spectrogram(audio,
                        n_fft=4096,
                        hop_length=512):
    """
    Compute magnitude spectrogram in dB.
    """

    stft = librosa.stft(
        audio,
        n_fft=n_fft,
        hop_length=hop_length
    )

    magnitude = np.abs(stft)

    spectrogram_db = librosa.amplitude_to_db(
        magnitude,
        ref=np.max
    )

    return spectrogram_db


def find_peaks(
    spectrogram,
    neighborhood_size=20,
    amp_min=-40
):
    """
    Find local maxima in spectrogram.
    """

    local_max = maximum_filter(
        spectrogram,
        size=neighborhood_size
    )

    peaks_mask = (
        (spectrogram == local_max)
        & (spectrogram > amp_min)
    )

    freq_idx, time_idx = np.where(peaks_mask)

    peaks = list(
        zip(time_idx, freq_idx)
    )

    return peaks


def generate_hashes(
    peaks,
    fan_value=10
):
    """
    Create Shazam-style hashes.

    Hash format:
    (freq1, freq2, delta_time)
    """

    hashes = []

    peaks = sorted(peaks)

    for i in range(len(peaks)):

        t1, f1 = peaks[i]

        for j in range(
            1,
            fan_value + 1
        ):

            if i + j >= len(peaks):
                break

            t2, f2 = peaks[i + j]

            dt = t2 - t1

            if dt <= 0:
                continue

            h = (
                int(f1),
                int(f2),
                int(dt)
            )

            hashes.append(
                (h, t1)
            )

    return hashes


def fingerprint_song(
    file_path
):
    """
    Complete pipeline.
    """

    audio, sr = load_audio(
        file_path
    )

    spec = compute_spectrogram(
        audio
    )

    peaks = find_peaks(
        spec
    )

    hashes = generate_hashes(
        peaks
    )

    return {
        "spectrogram": spec,
        "peaks": peaks,
        "hashes": hashes
    }

