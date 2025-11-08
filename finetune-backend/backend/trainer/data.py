from datasets import load_dataset, Dataset, DatasetDict
from typing import Optional, Dict, Any
import requests, tempfile
import os

def load_any_dataset(
    dataset_name_or_path: str,
    split_map: Optional[Dict[str, str]] = None,
    field_map: Optional[Dict[str, str]] = None,
    streaming: bool = False,
    **kw
) -> DatasetDict:
    """
    - If HF hub name (e.g., 'mlabonne/FineTome-100k'), loads from hub.
    - If local path to JSONL/CSV/JSON, builds DatasetDict with 'train'.
    - field_map lets you map { "prompt": "input", "completion": "output" } etc.
    """
    if os.path.isdir(dataset_name_or_path) or os.path.isfile(dataset_name_or_path):
        path = dataset_name_or_path
        if path.endswith(".jsonl") or path.endswith(".json"):
            ds = load_dataset("json", data_files={"train": path}, split=None)
        elif path.endswith(".csv"):
            ds = load_dataset("csv", data_files={"train": path}, split=None)
        else:
            raise ValueError("Unsupported local dataset format. Use .jsonl, .json, or .csv")
    elif dataset_name_or_path.startswith("http"):
        r = requests.get(dataset_name_or_path)
        r.raise_for_status()
        suffix = ".jsonl" if dataset_name_or_path.endswith(".jsonl") else ".json"
        tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
        tmp.write(r.content)
        tmp.close()
        dataset_name_or_path = tmp.name

    if isinstance(ds, Dataset):
        ds = DatasetDict({"train": ds})

    if field_map:
        def mapper(ex):
            mapped = {}
            for new, old in field_map.items():
                mapped[new] = ex.get(old, "")
            return mapped
        ds = ds.map(mapper)

    return ds
