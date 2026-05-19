CONFIG_240M = {
    "vocab_size": 32000,
    "num_hidden_layers": 12,
    "hidden_size": 1024,
    "num_attention_heads": 8,
    "num_key_value_heads": 8,
    "intermediate_size": 4096,
    "max_position_embeddings": 2048,
    "rms_norm_eps": 1e-5,
    "rope_theta": 10000.0,
}

CONFIG_700M = {
    "vocab_size": 32000,
    "num_hidden_layers": 32,
    "hidden_size": 2048,
    "num_attention_heads": 16,
    "num_key_value_heads": 8,
    "intermediate_size": 5632,
    "max_position_embeddings": 4096,
    "rms_norm_eps": 1e-5,
    "rope_theta": 10000.0,
}

CONFIG_1B = {
    "vocab_size": 32000,
    "num_hidden_layers": 32,
    "hidden_size": 2048,
    "num_attention_heads": 16,
    "num_key_value_heads": 8,
    "intermediate_size": 8192,
    "max_position_embeddings": 4096,
    "rms_norm_eps": 1e-5,
    "rope_theta": 10000.0,
}


def get_config(model_size: str = "1b") -> dict:
    if model_size.lower() == "240m":
        return CONFIG_240M.copy()
    elif model_size.lower() == "700m":
        return CONFIG_700M.copy()
    return CONFIG_1B.copy()