from dataclasses import dataclass


@dataclass
class DatasetConfig:
    name: str
    hf_id: str
    purpose: str
    tokens: int
    mix_percent: float


PRETRAIN_DATASETS = [
    DatasetConfig("FineWeb-Edu", "HuggingFaceFW/fineweb-edu", "High-quality educational web text", 2_100_000_000, 35.0),
    DatasetConfig("Stack v2 (filtered)", "bigcode/the-stack-v2-train-smol-ids", "Code across 600+ languages", 1_200_000_000, 20.0),
    DatasetConfig("Wikipedia EN", "wikimedia/wikipedia", "Factual grounding", 600_000_000, 10.0),
    DatasetConfig("OpenWebMath", "open-web-math/open-web-math", "Mathematical reasoning", 600_000_000, 10.0),
    DatasetConfig("Books3 subset", "the_pile", "Long-form reasoning", 900_000_000, 15.0),
    DatasetConfig("Dolma CC (filtered)", "allenai/dolma", "Diverse general knowledge", 600_000_000, 10.0),
]

TARGET_PRETRAIN_TOKENS = 6_000_000_000
INSTRUCTION_EXAMPLES = 120_000
TRAIN_EVAL_SPLIT = 0.9
VOCAB_SIZE = 32000
MAX_SEQ_LENGTH = 4096
MIN_TOKENS = 64
MAX_TOKENS = 4096