
import pickle
from collections import Counter

from src.fingerprint import fingerprint_song


class SongMatcher:

    def __init__(
        self,
        db_path="database/fingerprints.pkl"
    ):

        with open(
            db_path,
            "rb"
        ) as f:

            self.database = pickle.load(f)

    def identify(
        self,
        query_file
    ):

        query = fingerprint_song(
            query_file
        )

        offsets = Counter()

        for h, query_offset in query["hashes"]:

            if h not in self.database:
                continue

            matches = self.database[h]

            for song_name, db_offset in matches:

                delta = (
                    db_offset
                    - query_offset
                )

                offsets[
                    (
                        song_name,
                        delta
                    )
                ] += 1

        if len(offsets) == 0:

            return {
                "song": None,
                "score": 0
            }

        best_match = offsets.most_common(1)[0]

        song_name = best_match[0][0]
        score = best_match[1]

        return {
            "song": song_name,
            "score": score,
            "offsets": offsets
        }

