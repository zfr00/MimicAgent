"""FlashAttention-2 selection for Megatron training actors."""

from importlib.metadata import PackageNotFoundError, version
import logging
import os

TRAIN_ENV = {'NVTE_FLASH_ATTN': '1', 'NVTE_FUSED_ATTN': '0', 'NVTE_UNFUSED_ATTN': '0'}
FA2_VERSION = '2.8.3'


def check_flash_attention(*, training=False):
    try:
        installed = version('flash-attn').split('+', 1)[0]
    except PackageNotFoundError:
        installed = None
    if installed != FA2_VERSION:
        raise RuntimeError(f'Training requires flash-attn=={FA2_VERSION}; found {installed}')
    for package in ('flash-attn-3', 'flash-attn-4'):
        try:
            version(package)
        except PackageNotFoundError:
            continue
        raise RuntimeError(f'Remove {package} from the ARIS training environment; see train/Dockerfile')
    if training:
        if any(os.environ.get(key) != value for key, value in TRAIN_ENV.items()):
            raise RuntimeError('Training attention environment must select FlashAttention-2')
        from transformer_engine.pytorch.attention.dot_product_attention import backends
        if backends.flash_attn_func is None:
            raise RuntimeError('FlashAttention-2 unavailable in TransformerEngine')
        if backends.flash_attn_func_v3 is not None or backends.flash_attn_func_v4 is not None:
            raise RuntimeError('TransformerEngine loaded another FlashAttention version')
        logging.getLogger(__name__).info('Training attention: FlashAttention-2 %s', installed)
