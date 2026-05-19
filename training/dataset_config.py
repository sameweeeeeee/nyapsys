from dataclasses import dataclass


@dataclass
class DatasetConfig:
    name: str
    hf_id: str
    purpose: str
    mix_percent: float


DATASET_MIX = [
    DatasetConfig(
        name="FineWeb",
        hf_id="HuggingFaceFW/fineweb",
        purpose="General knowledge pretraining",
        mix_percent=40.0,
    ),
    DatasetConfig(
        name="Wikipedia EN",
        hf_id="wikimedia/wikipedia",
        purpose="Factual knowledge",
        mix_percent=20.0,
    ),
    DatasetConfig(
        name="Books3",
        hf_id="the_pile",
        purpose="Long-form reasoning",
        mix_percent=15.0,
    ),
    DatasetConfig(
        name="Alpaca Cleaned",
        hf_id="yahma/alpaca-cleaned",
        purpose="Instruction following",
        mix_percent=10.0,
    ),
    DatasetConfig(
        name="SQuAD v2",
        hf_id="rajpurkar/squad_v2",
        purpose="Document Q&A",
        mix_percent=8.0,
    ),
    DatasetConfig(
        name="Custom JSONL",
        hf_id="local",
        purpose="Nyapsys-specific examples",
        mix_percent=7.0,
    ),
]

TARGET_PRETRAIN_TOKENS = 14_000_000_000
INSTRUCTION_EXAMPLES = 50000
TRAIN_EVAL_SPLIT = 0.9

VOCAB_SIZE = 32000
MAX_SEQ_LENGTH = 4096