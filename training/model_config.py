CONFIG_240M_DENSE = {
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

CONFIG_1B_DENSE = {
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

CONFIG_2B_MOE = {
    "vocab_size": 32000,
    "num_hidden_layers": 32,
    "hidden_size": 2048,
    "num_attention_heads": 16,
    "num_key_value_heads": 8,
    "max_position_embeddings": 4096,
    "rms_norm_eps": 1e-5,
    "rope_theta": 10000.0,
    "num_experts": 4,
    "num_experts_per_token": 2,
    "expert_intermediate_size": 4096,
    "router_aux_loss_coef": 0.01,
}
# Total params: ~2B. Active per token: ~1B. VRAM during training: ~9–11GB.


def get_config(model_size: str = "2b_moe") -> dict:
    if model_size.lower() == "240m":
        return CONFIG_240M_DENSE.copy()
    elif model_size.lower() == "1b":
        return CONFIG_1B_DENSE.copy()
    return CONFIG_2B_MOE.copy()