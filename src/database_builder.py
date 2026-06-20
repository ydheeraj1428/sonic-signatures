import os
import pickle

from fingerprint import fingerprint_song

SONGS_DIR = "songs"
OUTPUT_DB = "database/fingerprints.pkl"

database = {}

files = [
    f for f in os.listdir(SONGS_DIR)
    if f.endswith(".mp3")
]

print(f"\nFound {len(files)} songs\n")

for i, song in enumerate(files):

    path = os.path.join(
        SONGS_DIR,
        song
    )

    print(
        f"[{i+1}/{len(files)}] Processing {song}"
    )

    result = fingerprint_song(path)

    for h, offset in result["hashes"]:

        if h not in database:
            database[h] = []

        database[h].append(
            (
                song,
                offset
            )
        )

with open(
    OUTPUT_DB,
    "wb"
) as f:

    pickle.dump(
        database,
        f
    )

print("\nDatabase saved")
print(
    f"Total unique hashes: {len(database)}"
)