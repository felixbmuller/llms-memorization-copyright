"""Apply the fuzzy matching to raw model outputs."""

import csv
import os
from pathlib import Path
import random
from typing import Dict

from tqdm import tqdm
from fire import Fire

from src.text import SeqMatch, find_common_subsequences, load_ebook, process_text
from src.utils import BASE_PATH, load_jsonl, load_json
from src.prompt_specific_processing import prompt_specific_decoding, clean_match

# column definitions for labelling output
rows_per_output = [
    "book_id",
    "prompt_id",
    "label",
    "highest_char_count",
    "model_output",
    "matches",
    "final_prompt",
]

# column definitions for evaluation output
rows_per_match = [
    "book_id",
    "prompt_id",
    "word_count",
    "char_count",
    "skipped_chars_model",
    "skipped_chars_book",
    "chapter",
    "match_text",
]


def _most_recent_mdate(files):

    mdate = 0

    for path in files:
        if os.path.isfile(path):
            this_mdate = os.path.getmtime(path)

            if this_mdate > mdate:
                mdate = this_mdate

    return mdate


def main(out_file_name, *model_files, corpus="publicdomain", mode="match", min_length=None):
    """Perform fuzzy test matching either for the purpose of automatic evaluation (mode="match") or manual output labelling (mode="output").

    Parameters
    ----------
    out_file_name : str
        base name of the output, should equal the folder name of the raw input
    select : str, optional
        corpus "copyright" or "publicdomain", by default "publicdomain"
    mode : str, optional
        matching mode ("match" or "output"), by default "match"
    min_length : int, optional
        minimum match length in words for a match to be counted, by default 8 if mode="match" else 5. Less than 5 yields too many false positives.

    Raises
    ------
    ValueError
        Illegal arguments
    """
    rows_list_per_output = []
    rows_list_per_match = []

    if corpus == "publicdomain":
        book_files = load_json(f"{BASE_PATH}/data/books_publicdomain.json")
    elif corpus == "copyright":
        book_files = load_json(f"{BASE_PATH}/data/books_copyright.json")

    print(out_file_name)

    full_output_name = f"{out_file_name}_corpus={corpus}.csv"

    if os.path.exists(full_output_name) and os.path.getmtime(full_output_name) > _most_recent_mdate(
        model_files
    ):
        print("  SKIP")
        return

    for file in model_files:
        file = Path(file)
        book_id = file.stem

        if book_id not in book_files:
            continue

        if min_length is None:
            min_length = 8 if mode == "match" else 5
            # less than five yields too many false positives

        print(f"Processing {file}")
        subseq_ret, processed_outputs = perform_matching(file, book_files[book_id], min_length)
        if mode == "match":
            create_rows_per_match(subseq_ret, book_id, rows_list_per_match)
        elif mode == "output":
            create_rows_per_output(subseq_ret, processed_outputs, book_id, rows_list_per_output)
        else:
            raise ValueError("illegal mode")

    if mode == "output":
        with open(full_output_name, "w", newline="") as csvfile:
            writer = csv.writer(csvfile, quoting=csv.QUOTE_MINIMAL)
            writer.writerow(rows_per_output)

            random.Random(42).shuffle(rows_list_per_output)
            writer.writerows(rows_list_per_output)

    if mode == "match":
        with open(full_output_name, "w", newline="") as csvfile:
            writer = csv.writer(csvfile, quoting=csv.QUOTE_MINIMAL)
            writer.writerow(rows_per_match)

            writer.writerows(rows_list_per_match)


def perform_matching(model_output_file, book_file, min_length):
    """Perform fuzzy matching for a specific raw model output file (jsonl) and book file.

    Parameters
    ----------
    model_output_file : str
        jsonl file to process
    book_file : str
        book txt to process
    min_length : int
        min match length in words

    """
    chapters, meta = load_ebook(BASE_PATH + book_file)

    model_output = load_jsonl(model_output_file)

    processed_outputs = {}

    for e in model_output:
        out = prompt_specific_decoding(e["prompt_id"], e["output"], meta)
        processed_outputs[e["prompt_id"]] = (
            e["final_prompt"],
            e["output"],
            process_text(out),
        )

    subseq_ret: Dict[str, list[SeqMatch]] = {
        prompt_id: [
            m
            for m in find_common_subsequences(short, chapters, min_length=min_length, padding=0)
            if not clean_match(m, prompt_id, meta)
        ]
        for prompt_id, (_, _, short) in tqdm(processed_outputs.items())
    }

    return subseq_ret, processed_outputs


def create_rows_per_match(subseq_ret, book_id, rows_list):
    """Produce rows in the output csv based on the output of `perform_matching`"""
    for prompt_id, seq_result in subseq_ret.items():
        for match in seq_result:
            rows_list.append(
                [
                    book_id,
                    prompt_id,
                    match.word_count,
                    match.char_count,
                    match.skipped_short,
                    match.skipped_long,
                    match.chapter,
                    match.text,
                ]
            )


def create_rows_per_output(subseq_ret, processed_outputs, book_id, rows_list):
    """Produce rows in the output csv based on the output of `perform_matching`"""
    for prompt_id, seq_result in subseq_ret.items():
        final_prompt, model_output, _ = processed_outputs[prompt_id]
        max_char_count = max((match.char_count for match in seq_result), default=0)
        matches = "\n".join(match.text for match in seq_result)

        rows_list.append(
            [
                book_id,
                prompt_id,
                "",
                max_char_count,
                model_output,
                matches,
                final_prompt,
            ]
        )


if __name__ == "__main__":
    Fire(main)
