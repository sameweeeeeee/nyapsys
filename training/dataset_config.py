from dataclasses import dataclass
from typing import Optional


@dataclass
class DatasetConfig:
    name: str
    hf_id: str
    purpose: str
    mix_percent: float


DATASET_MIX = [
    DatasetConfig(
        name="Alpaca Cleaned",
        hf_id="yahma/alpaca-cleaned",
        purpose="General instruction following",
        mix_percent=30.0,
    ),
    DatasetConfig(
        name="SQuAD v2",
        hf_id="rajpurkar/squad_v2",
        purpose="Reading comprehension / document Q&A",
        mix_percent=25.0,
    ),
    DatasetConfig(
        name="LLaVA Instruct",
        hf_id="HuggingFaceH4/llava-instruct-mix-vsft",
        purpose="Vision + image Q&A — critical for 11B Vision",
        mix_percent=25.0,
    ),
    DatasetConfig(
        name="LMSYS Chat 1M",
        hf_id="lmsys/lmsys-chat-1m",
        purpose="Real multi-turn conversations",
        mix_percent=15.0,
    ),
    DatasetConfig(
        name="Custom JSONL",
        hf_id="local",
        purpose="Nyapsys persona + domain-specific examples",
        mix_percent=5.0,
    ),
]

TARGET_TRAIN_SAMPLES = 50000
TRAIN_EVAL_SPLIT = 0.9

CHAT_TEMPLATE = "<|begin_of_text|><|start_header_id|>{role}<|end_header_id|>\n{content}<|eot_id|>"

SYSTEM_MESSAGE = "You are Nyapsys, a helpful AI assistant. You answer questions accurately, read and analyse files, and understand images. Be concise but thorough. If you are unsure, say so."

VISION_SYSTEM_INJECT = "You are Nyapsys. "

MAX_SEQ_LENGTH = 2048