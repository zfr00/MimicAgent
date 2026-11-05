from functools import lru_cache


@lru_cache(maxsize=1)
def _provider():
    from .attention import check_flash_attention
    check_flash_attention(training=True)
    from megatron.training import get_args
    from megatron.bridge import AutoBridge
    from megatron.core.transformer.enums import AttnBackend
    from slime.utils.megatron_bridge_utils import patch_auto_bridge_hf_config
    import slime_plugins.megatron_bridge  # noqa: F401, registers the pinned bridges

    args = get_args()
    bridge = patch_auto_bridge_hf_config(AutoBridge.from_hf_pretrained(args.hf_checkpoint, trust_remote_code=True))
    provider = bridge.to_megatron_provider(load_weights=False)
    for key in ("tensor_model_parallel_size", "pipeline_model_parallel_size", "expert_model_parallel_size",
                "expert_tensor_parallel_size", "sequence_parallel", "context_parallel_size", "variable_seq_lengths",
                "recompute_granularity", "recompute_method", "recompute_num_layers", "attention_dropout",
                "hidden_dropout"):
        if not hasattr(provider, key):
            raise RuntimeError(f"Installed Megatron Bridge lacks required provider setting: {key}")
        setattr(provider, key, getattr(args, key))
    for key in ("freeze_language_model", "freeze_vision_model", "freeze_vision_projection"):
        if hasattr(provider, key):
            setattr(provider, key, False)
    provider.attention_backend = (AttnBackend[args.attention_backend]
                                  if isinstance(args.attention_backend, str) else args.attention_backend)
    provider.finalize()
    return provider


def model_provider(pre_process=True, post_process=True, vp_stage=None):
    return _provider().provide(pre_process=pre_process, post_process=post_process, vp_stage=vp_stage)
