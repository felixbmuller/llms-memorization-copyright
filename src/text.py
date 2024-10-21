"""Contains text preprocessing and the fuzzy matching code."""

from collections import defaultdict
from dataclasses import dataclass
from typing import Optional
import string
import json

from unidecode import unidecode


def load_ebook(path):
    """Load an ebook file-

    Parameters
    ----------
    path : str
        path to ebook txt

    Returns
    -------
    list[str]
        text for all chapters in the ebooks
    dict[str, str]
        metadata of the ebook needed for prompt generation
    """
    with open(path, encoding="utf-8-sig") as fp:
        text = fp.read()

    meta_str, text = text.split("###END METADATA###", 1)

    meta_raw = json.loads(meta_str)

    meta = {k: v for k, v in meta_raw.items() if not isinstance(v, list)}

    for k, v in meta_raw.items():
        if isinstance(v, list):
            meta.update({f"{k}{idx}": v[idx] for idx in range(len(v))})

    chapters = text.split("###CHAPTER###")

    text = [process_text(c) for c in chapters]

    return text, meta


def process_text(s: str) -> list[str]:
    """Perform all relevant preprocessing steps, e.g. removing special characters, lowercase;
    tokenize at whitespace.

    Parameters
    ----------
    text : str
        _description_

    Returns
    -------
    List[str]
        _description_
    """
    s = s.translate(str.maketrans({"’": None, "'": None, "”": None, "“": None}))
    s = unidecode(s)
    s = s.translate(str.maketrans(string.punctuation, " " * len(string.punctuation)))
    s = s.translate(str.maketrans(string.whitespace, " " * len(string.whitespace)))
    s = s.casefold()

    # Split and filter empty strings
    parts = [part for part in s.split(" ") if part]

    return parts


@dataclass
class SeqMatch:
    text: str  # printable representation
    short_text: str  # exactly the matched short text
    long_text: str  # excactly the matched long text
    word_count: int
    char_count: int
    sequence: "Sequence"
    skipped_short: int  # Number of characters per match that were skipped in the model output
    skipped_long: int  # Number of characters per match that were skipped in the original text
    chapter: int = -1


class Sequence:
    elements: list[tuple[int, int]]

    def __init__(self, base_seq: Optional["Sequence"], short_idx: int, long_idx: int):
        """Create a new match by adding one more word to `base_seq`. A word is represented by both its index in the short sequence and the long sequence."""
        if base_seq is None:
            self.elements = [(short_idx, long_idx)]
        else:
            self.elements = [*base_seq.elements, (short_idx, long_idx)]

    def fuzzy_okay(self):
        """Returns True if it would be okay to append a fuzzily matching word to `self`, i.e. if
        the last two words of `self` were exact matches."""
        if len(self) < 2:
            return False

        prev_short, prev_long = self.elements[-1]
        pprev_short, pprev_long = self.elements[-2]

        # Last two matched words were exact
        is_okay = prev_short == pprev_short + 1 and prev_long == pprev_long + 1

        return is_okay

    def __len__(self):
        return len(self.elements)

    def __contains__(self, subseq):
        if len(subseq) >= len(self):
            return False

        return set(subseq.elements).issubset(self.elements)

    def prev(self):
        return self.elements[-2]

    def diff(self, short: list[str], long: list[str], padding=5):
        """Produce all information we need to process this match downstream, e.g. character count and human-readable representation."""
        short_exact_start, long_exact_start = self.elements[0]
        short_exact_end, long_exact_end = self.elements[-1]

        short_start = max(0, short_exact_start - padding)
        long_start = max(0, long_exact_start - padding)
        short_end = min(len(short), short_exact_end + padding + 1)
        long_end = min(len(long), long_exact_end + padding + 1)

        short_text = " ".join(short[short_exact_start : short_exact_end + 1])
        long_text = " ".join(long[long_exact_start : long_exact_end + 1])

        short_subseq = short[short_start:short_end]
        long_subseq = long[long_start:long_end]

        short_idxs, long_idxs = zip(*self.elements)

        short_casing = " ".join(
            [
                (s.upper() if curr_idx in short_idxs else s)
                for curr_idx, s in enumerate(short_subseq, short_start)
            ]
        )
        long_casing = " ".join(
            [
                (s.upper() if curr_idx in long_idxs else s)
                for curr_idx, s in enumerate(long_subseq, long_start)
            ]
        )

        word_count = len(self.elements)
        char_count = sum(1 for c in short_casing if c.isupper())
        assert char_count == sum(1 for c in long_casing if c.isupper()), short_casing + long_casing

        if padding == 0:
            skipped_short = sum(1 for c in short_casing if c.islower())
            skipped_long = sum(1 for c in long_casing if c.islower())
        else:
            skipped_short = skipped_long = -1

        char_count += word_count - 1  # whitespace

        s = f"INPUT: {short_casing}\nREFER: {long_casing}\nWORDS: {word_count:3} CHARS: {char_count:4}\n"

        return SeqMatch(
            s,
            short_text,
            long_text,
            word_count,
            char_count,
            self,
            skipped_short,
            skipped_long,
        )


class SequenceMap:
    """This is a map of all matches between two sequences. It is build iteratively by calling `add()` for each word in the shorter sequence `short`."""

    elements: list[dict[int, Sequence]]

    def __init__(self) -> None:
        self.elements = []

    def add(self, match_idxs: list[int]):
        """For the current word in `short`, process all matches with words from `long`. Matches are represented as the index in `long` where they appear."""
        prev = self.elements[-1] if len(self.elements) > 0 else {}
        pprev = self.elements[-2] if len(self.elements) > 1 else {}

        current = {}

        short_idx = len(self.elements)  # index of this short word

        for match_idx in match_idxs:
            # Case 1: Perfect match
            if match_idx - 1 in prev.keys():
                current[match_idx] = Sequence(prev[match_idx - 1], short_idx, match_idx)

                # Do not allow fuzzy matches on a sequence if it could also be a perfect match
                del prev[match_idx - 1]

            # Case 2: Skipping one word in `short`
            elif match_idx - 2 in pprev.keys() and pprev[match_idx - 2].fuzzy_okay():
                current[match_idx] = Sequence(pprev[match_idx - 2], short_idx, match_idx)

            # Case 3: Skipping one word in `long`
            elif match_idx - 2 in prev.keys() and prev[match_idx - 2].fuzzy_okay():
                current[match_idx] = Sequence(prev[match_idx - 2], short_idx, match_idx)

            # Case 4: One differing word in `long` and `short`
            elif match_idx - 1 in pprev.keys() and pprev[match_idx - 1].fuzzy_okay():
                current[match_idx] = Sequence(pprev[match_idx - 1], short_idx, match_idx)

            # Case 5: Cannot continue match, create new one
            else:
                current[match_idx] = Sequence(None, short_idx, match_idx)

        self.elements.append(current)

    def prune(self, min_length: int = 4):
        """Delete all matches which are too short to be included in the analysis"""
        for current in self.elements:
            for match_idx in list(current.keys()):
                if len(current[match_idx]) < min_length:
                    del current[match_idx]

            for match_idx, seq in current.items():
                prev_short_idx, prev_long_idx = seq.prev()
                self.elements[prev_short_idx].pop(prev_long_idx, None)  # delete if exists

    def get_sequences(self, deduplicate=True) -> list[Sequence]:
        """Get all sequences that are left after pruning. Deduplicate identical matches"""
        seqs = []

        for current in self.elements:
            seqs += list(current.values())

        if deduplicate:
            seqs = [seq for seq in seqs if not any(seq in other_seq for other_seq in seqs)]

        return seqs


def find_common_subsequences(short: list[str], chapters: list[list[str]], min_length=4, padding=5):
    """Apply fuzzy common subsequence matching. It is applied chapter-wise so that we can identify in which chapter a match occurred."""
    all_matches = []

    for cidx, c in enumerate(chapters, start=1):
        matches = _find_common_subsequences(short, c, min_length=min_length, padding=padding)

        for match in matches:
            match.chapter = cidx

        all_matches += matches

    return all_matches


def _find_common_subsequences(
    short: list[str], long: list[str], min_length=4, padding=5
) -> list[SeqMatch]:
    """Perform the actual matching"""

    # shord word to index lookup
    short_word2index = defaultdict(list)
    for idx, word in enumerate(short):
        short_word2index[word].append(idx)

    # Collect all matching indices for each word in short
    matching_idxs = [[] for _ in range(len(short))]
    for long_idx, word in enumerate(long):
        for short_idx in short_word2index[word]:
            matching_idxs[short_idx].append(long_idx)

    seq_map = SequenceMap()

    # Iteratively build matching sequences from single-word matches
    for match_idxs in matching_idxs:
        seq_map.add(match_idxs)

    seq_map.prune(min_length=min_length)

    seqs = seq_map.get_sequences()

    diff = [seq.diff(short, long, padding=padding) for seq in seqs]

    return diff
