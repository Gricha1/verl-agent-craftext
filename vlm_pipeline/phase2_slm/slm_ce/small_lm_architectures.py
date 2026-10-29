"""Random-init Qwen3 models matched to compact GPT-2 parameter budgets."""
from transformers import Qwen3Config, Qwen3ForCausalLM


def matched_qwen3(config, *, target_parameters):
    """Keep depth/width/context/vocabulary; choose the closest 16-wide SwiGLU FFN.

    Half as many KV heads as query heads, tied embeddings, and no pretrained
    weights. n_positions is an alias for existing ZeroHero rollout adapters.
    """
    width, layers, heads = config.n_embd, config.n_layer, config.n_head
    if heads % 2 or width % heads:
        raise ValueError('Expected an even head count dividing the hidden width')
    head_dim, kv = width // heads, heads // 2
    fixed = config.vocab_size * width + width + layers * (
        2 * width * width + 2 * width * kv * head_dim + 2 * width + 2 * head_dim)
    intermediate = max(16, round((target_parameters - fixed) / (3 * width * layers * 16)) * 16)
    q = Qwen3Config(
        vocab_size=config.vocab_size, hidden_size=width, intermediate_size=intermediate,
        num_hidden_layers=layers, num_attention_heads=heads, num_key_value_heads=kv,
        head_dim=head_dim, max_position_embeddings=config.n_positions,
        n_positions=config.n_positions, rope_theta=10000., rms_norm_eps=1e-6,
        tie_word_embeddings=True, attention_bias=False, attention_dropout=0.,
        use_sliding_window=False, sliding_window=None,
        bos_token_id=config.bos_token_id, eos_token_id=config.eos_token_id,
        pad_token_id=config.pad_token_id, use_cache=True,
    )
    model = Qwen3ForCausalLM(q)
    actual = sum(p.numel() for p in model.parameters())
    if abs(actual / target_parameters - 1) > .01:
        raise ValueError(f'Parameter mismatch: {actual} vs {target_parameters}')
    model.config.comparison_target_parameters = target_parameters
    model.config.comparison_actual_parameters = actual
    return model
