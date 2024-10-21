# LLMs and Memorization: On Quality and Specificity of Copyright Compliance

**Felix B Mueller, Rebekka Görge, Anna K Bernzen, Janna C Pirk, Maximilian Poretschkin**

[[PDF]](https://arxiv.org/pdf/2405.18492) [[arXiv]](https://arxiv.org/abs/2405.18492)

This is the official repository for our [AIES 2024](https://www.aies-conference.com/2024/) paper.

## Setup

Create a virtual environment with Python 3.9.2. and run

```
pip install -r requirements.txt
pip install -e .
```

Rename `api_keys.py.template` to `api_keys.py` and insert API keys for OpenAI, TogetherAI, and AlephAlpha.

## Evaluate an LLM

### Query an LLM based on prompt templates

Run 

```bash
python src/query_llms.py --corpus=publicdomain together meta-llama/Meta-Llama-3-8B-Instruct-Turbo output/llama
```

to query LLama for all prompts (i.e. all prompt templates for all books) and  apply the fuzzy matching on both the copyright and public domain corpus.

Re-running `src/query_llms.py` with exactly the same parameters will only perform requests that previously failed. You should re-run it until no new requests are performed, to ensure that there is an output recorded for every prompt. Do not run an identical `query_llms` command twice at the same time, as this may cause duplicate model outputs that need to be filtered before evaluation. 


| Model Provider | Model Name Scheme |
|------------------|-------------|
| `openai` | gpt-4o, gpt-4-turbo, gpt-3.5-turbo, , ... |
| `together` | meta-llama/Meta-Llama-3-8B-Instruct-Turbo, togethercomputer/llama-2-70b-chat, ... |
| `alephalpha` | luminous-supreme-control, ... |

This will create a jsonl-file for each book in `output/llama`, which contains detailed information on the LLM outputs. 

### Find potenial copyright violations

Use

```bash
python src/matching.py --corpus=publicdomain output/llama_matching output/llama/*
```

to apply our fuzzy matching algorithm for finding potential copyright violations. It will compare the model outputs for each book against the original text of that book and report all fuzzy matches longer than 8 words (configure using `--min_length`). Our fuzzy matching algorithm allows one word deviations between model output and reference text. It finds 50% more matches than exact matching and does so without false positives (in our experiments, when setting a minimum match length of 160 characters and ensuring that book files do not contain low-entropy noise like licensing information). The most common reason for those one word deviations are differences between British English and American English and slight changes in wording between different editions. By using fuzzy matching, we can reliably detect copyright violations, even if a model was trained on a different edition than we are evaluating on.

This creates a csv-file `output/llama_matching_corpus=publicdomain.csv`, which contain one line per match between model outputs and book text. Each line contains:
- `book_id` unique book id, see `data/books_XXX.json`
- `prompt_id` unique prompt id, see `data/prompt_templates.json`
- `word_count` number of words in the match (only counting matched words, not those skipped because of the fuzzy criterion)
- `char_count` number of characters in the match (only counting matched words), can be used to evaluate the amount of text reproduction 
- `skipped_chars_model`, `skipped_chars_book` number of non-matching characters that are present in the fuzzy match, can be used to evaluate the fuzziness of the match
- `chapter` chapter of the book the match appeared in (based on the `###CHAPTER###` markers in the books)
- `match_text` human-readable representation of the match, including the corresponding text portions from model output and book file (matching parts are uppercase)

*If you want to perform manual labelling of output categories (Match, Hallucination, Refusal, ...), use `src/matching.py --mode=output ...` to create csv-files with one model output per line that are optimized for human labelling.*

Using the list of matches, you can calculate the significant reproduction rate score for the LLama model

```python
import pandas as pd

pd_df = pd.read_csv('output/llama_matching_corpus=publicdomain.csv')

srr_pd = pd_df.char_count[pd_df.char_count > 160].sum() / 20 #books
```

## Evaluate an LLM on copyrighted books

To properly evaluate the copyright compliance of a LLM, you need to compare its behavior on copyrighted and public domain books. 

Acquire the copyrighted books and add them to `data/books_copyright/`. See `data/books_copyright.json` for the expected final list of files and `data/books_public_domain/` for the file format. 

Run 

```bash
python src/query_llms.py --corpus=copyright together meta-llama/Meta-Llama-3-8B-Instruct-Turbo output/llama
python src/matching.py --select=copyright output/llama_matching output/llama/*
```

to generate a second csv-file `output/llama_matching_corpus=copyright.csv`.

You can now calculate copyright discrimination ratio and the SRR on both corpora using

```python
import pandas as pd

pd_df = pd.read_csv('output/llama_matching_corpus=publicdomain.csv')
cr_df = pd.read_csv('output/matching/llama_matching_corpus=copyright.csv')

srr_pd = pd_df.char_count[pd_df.char_count > 160].sum() / 20 #books
srr_cr = cr_df.char_count[cr_df.char_count > 160].sum() / 20 #books
cdr = srr_cr/srr_pd
```

## Custom experiments

### Multiple Trials

To run multiple trials with one model, simply create one output directory per trial, e.g. `output/llama_run=01`. You can run `src/query_llms.py` for multiple trials in parallel, but do not start two jobs for the same trial. In our paper, we report the average over 30 trials for most experiments.

### Other Models

You can test all models available via the APIs of OpenAI, TogetherAI and AlephAlpa. Most models should work out of the box. In case they do not, you may want to overwrite the `request()` function in `src/models.py`. There you can control how prompts are passed to models. 

### Other Prompts

Prompts are supplied as a json-encoded dictionary mapping unique prompt ids to prompt templates. The prompt template may contain placeholder of the form "{name}", where `name` must be present in a books metadata. If it is not present for a book, (i.e. main characters in non-fiction), the prompt template is skipped for that book. Prompt templates are stored in `data/prompt_templates.json`

If you want to use your own prompt templates, you may need to adjust the functions in `src/prompt_specific_processing.py` to ensure that your custom prompts are handled correctly. This is necessary if you want to perform text replacements before fuzzy matching (e.g. when trying to circumvent model filters with ROT-13) or if your prompt contains original text of the book (to exclude those parts from being counted as potential copyright violation).

### Custom Book Corpora

To use another book corpus, simply place your book files in `books_copyright` or `books_publicdomain` and change the corresponding `books_XXX.json` file. 

The books txt files are preprocessed to remove conversion errors that would prohibit string comparisons, e.g. sentences glued together. Junk information such as licensing, placeholder for images, etc. is removed. All book files start with a json-encoded dictionary of metadata, followed by an `###END METADATA###` marker. The text of the book may be separated into chapters by a `###CHAPTER###` marker, this is optional.
