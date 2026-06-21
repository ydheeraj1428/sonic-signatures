import matplotlib.pyplot as plt
import librosa
import librosa.display
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter


def plot_spectrogram(spec, title="Spectrogram"):

    fig, ax = plt.subplots(figsize=(10, 5))

    img = librosa.display.specshow(
        spec,
        x_axis="time",
        y_axis="hz",
        ax=ax
    )

    ax.set_title(title)

    fig.colorbar(
        img,
        ax=ax,
        format="%+2.0f dB"
    )

    return fig

def plot_constellation(
    spec,
    peaks,
    title="Constellation Map"
):

    fig, ax = plt.subplots(
        figsize=(10,5)
    )

    time_idx = [p[0] for p in peaks]
    freq_idx = [p[1] for p in peaks]

    ax.scatter(
        time_idx,
        freq_idx,
        s=2
    )

    ax.set_title(title)
    ax.set_xlabel("Time")
    ax.set_ylabel("Frequency Bin")

    return fig


def plot_offset_histogram(
    offsets,
    title="Offset Histogram"
):

    fig, ax = plt.subplots(
        figsize=(10, 4)
    )

    ax.hist(
        offsets,
        bins=50
    )

    ax.set_title(title)

    ax.set_xlabel(
        "Offset"
    )

    ax.set_ylabel(
        "Count"
    )

    return fig


def plot_offsets(offset_counter):

    top_offsets = offset_counter.most_common(50)

    if len(top_offsets) == 0:
        return None

    labels = [
        str(item[0][1])
        for item in top_offsets
    ]

    values = [
        item[1]
        for item in top_offsets
    ]

    fig, ax = plt.subplots(
        figsize=(8,4)
    )

    ax.bar(
        range(len(values)),
        values
    )

    ax.set_title(
        "Offset Match Histogram"
    )

    ax.set_xlabel(
        "Offset Difference"
    )

    ax.set_ylabel(
        "Votes"
    )

    return fig
