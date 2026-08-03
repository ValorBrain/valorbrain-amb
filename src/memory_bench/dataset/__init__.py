from .base import Dataset
from .beam import BEAMDataset
from .lifebench import LifeBenchDataset
from .locomo import LoComoDataset
from .longmemeval import LongMemEvalDataset
from .personamem import PersonaMemDataset
from .sdebench import SdebenchDataset

REGISTRY: dict[str, type[Dataset]] = {
    "beam":         BEAMDataset,
    "lifebench":    LifeBenchDataset,
    "locomo":       LoComoDataset,
    "longmemeval":  LongMemEvalDataset,    "personamem":   PersonaMemDataset,
    "sdebench":     SdebenchDataset,
}


def get_dataset(name: str) -> Dataset:
    if name not in REGISTRY:
        raise ValueError(f"Unknown dataset: '{name}'. Available: {list(REGISTRY)}")
    return REGISTRY[name]()
