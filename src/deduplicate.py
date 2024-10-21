"""Fix duplicate raw model outputs before running evaluations.

When performing model requests in parallel, it can happen that we record two model outputs for one request. To avoid issues with evaluation later on, those duplicates need to be removed.
"""

from glob import glob
import json
from pathlib import Path

import pandas as pd
from fire import Fire

from src.utils import load_jsonl


def print_duplicates(folder):
    """Scan the whole raw output folder for duplicate entries

    Parameters
    ----------
    folder : str
        path to raw output folder
    """
    for folder in glob(folder):
        name = Path(folder).name
        for file in glob(folder + "/*"):
            book_id = Path(file).stem.split("_")[-1]
            data = load_jsonl(file)
            prompt_ids = pd.Series([e["prompt_id"] for e in data]).value_counts()
            duplicates = prompt_ids[prompt_ids > 1]
            if not duplicates.empty:
                print(name, book_id)
                print(duplicates)


def deduplicate_file(file):
    """Deduplicate one book file of one model

    Parameters
    ----------
    file : str
        path to .jsonl file
    """
    prompt_ids = set()
    data = load_jsonl(file)
    out_data = []
    skipped = 0
    for elem in data:
        if elem["prompt_id"] not in prompt_ids:
            prompt_ids.add(elem["prompt_id"])
            out_data.append(elem)
        else:
            skipped += 1

    if skipped > 0:
        print(file.split("/")[-1], skipped)
        with open(file, "w") as fp:
            for elem in out_data:
                fp.write(json.dumps(elem))
                fp.write("\n")
        print(len(data), len(out_data))


if __name__ == "__main__":
    Fire({"scan": print_duplicates, "fix": deduplicate_file})
