"""Send prompts to LLMs and store results."""

import os

from fire import Fire
from tqdm import tqdm

from src.text import load_ebook
from src.utils import load_json, load_jsonl, BASE_PATH
from src.models import get_model


def main(
    provider,
    model_name,
    output_dir,
    corpus="publicdomain",
    temp=0.7,
    sleep_interval=1,
):
    """
    Send prompts to LLMs and store results.

    Parameters
    ----------
    provider : str
        provider of the LLM, e.g. together, openai
    model_name : str
        name of the model, e.g. gpt-3.5-turbo
    output_dir : str
        output directory for the results (one jsonl file per book)
    corpus : str, optional
        corpus to use, by default "publicdomain"
    temp : float, optional
        temperature for the LLM, by default 0.7
    sleep_interval : int, optional
        sleep interval between requests, by default 1 second. Adjust this to avoid rate limiting.
    """

    if corpus == "publicdomain":
        book_files = load_json(f"{BASE_PATH}/data/books_publicdomain.json")
    elif corpus == "copyright":
        book_files = load_json(f"{BASE_PATH}/data/books_copyright.json")

    prompts = load_json(f"{BASE_PATH}/data/prompt_templates.json")
    llm = get_model(provider, model_name, temp=temp, sleep_interval=sleep_interval)

    os.makedirs(output_dir, exist_ok=True)

    performed_requests, performed_errors = 0, 0

    with tqdm(total=len(prompts) * len(book_files)) as pbar:
        for book_id, book_path in book_files.items():
            this_success, this_errors = perform_query(
                llm, prompts, book_path, f"{output_dir}/{book_id}.jsonl", pbar=pbar
            )
            performed_requests += this_success
            performed_errors += this_errors

    print("New requests performed: ", performed_requests)
    print("Errors encountered: ", performed_errors)


def perform_query(llm, prompts, book_file, out_file, pbar):

    performed_requests, errors = 0, 0
    _, meta = load_ebook(book_file)

    try:
        prev_requests = load_jsonl(out_file)
        skip_prompt_ids = {r["prompt_id"] for r in prev_requests}
    except FileNotFoundError:
        skip_prompt_ids = set()

    with open(out_file, "a") as fp:

        for prompt_id, prompt in prompts.items():
            if prompt_id not in skip_prompt_ids:
                status = llm(prompt, prompt_id, meta, fp)
                if status == "performed":
                    performed_requests += 1
                elif status == "error":
                    errors += 1
            pbar.update(1)

    return performed_requests, errors


if __name__ == "__main__":
    Fire(main)
