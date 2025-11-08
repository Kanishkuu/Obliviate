from datasets import load_dataset, Dataset, DatasetDict
from typing import Optional, Dict, Any, Union
import json, csv, os

def load_any_dataset(
    dataset_name_or_path: str,
    split_map: Optional[Dict[str, str]] = None,
    field_map: Optional[Dict[str, str]] = None,
    streaming: bool=False,
    **kw
) -> DatasetDict:
    """
    - If HF hub name (e.g., 'mlabonne/FineTome-100k'), loads from hub.
    - If local path to JSONL/CSV, builds DatasetDict with 'train' split.
    - field_map lets you map { "input": "prompt", "target": "completion" } etc.
    """
    if os.path.isdir(dataset_name_or_path) or os.path.isfile(dataset_name_or_path):
        # infer format
        path = dataset_name_or_path
        if path.endswith(".jsonl"):
            ds = load_dataset("json", data_files={"train": path}, split=None)
        elif path.endswith(".json"):
            ds = load_dataset("json", data_files={"train": path}, split=None)
        elif path.endswith(".csv"):
            ds = load_dataset("csv", data_files={"train": path}, split=None)
        else:
            raise ValueError("Unsupported local dataset format. Use .jsonl, .json, or .csv")
    else:
        ds = load_dataset(dataset_name_or_path, streaming=streaming)

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
