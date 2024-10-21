from datetime import datetime
import json
from pathlib import Path

BASE_PATH = str(Path(__file__).parent.parent) + "/"


def load_json(path):
    """Load json directly from a path"""
    with open(path) as fp:
        return json.load(fp)


def timestamp():
    """Create a timestamp string"""
    return datetime.now().strftime("%Y-%m-%dT%H-%M-%S%Z")


def load_jsonl(path):
    """Load a jsonl file at once"""
    with open(path) as fp:
        lines = fp.readlines()

    content = [json.loads(line) for line in lines]

    return content
