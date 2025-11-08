from ..trainer.data import load_any_dataset
from typing import Optional, Dict

def load_dataset_any(dataset: str,
                     field_map: Optional[Dict[str, str]] = None,
                     streaming: bool = False):
    return load_any_dataset(dataset, field_map=field_map, streaming=streaming)
