"""v6/v7 → HF PreTrainedModel 最小适配器（供 lm_eval 加载）。
关键: lm_eval 的 loglikelihood 任务只需模型 forward 出标准 logits [B,T,V] + 因果性。
这里包一个 HF 兼容 wrapper，注册进 AutoModelForCausalLM，供 lm_eval --model hf 加载。
注意: 这是探路版——先验证 v6 因果性 + 能否出分, 不追求完整 generate 适配。
"""
import torch
import torch.nn as nn
from transformers import PreTrainedModel, PretrainedConfig, AutoModelForCausalLM


class V6HFConfig(PretrainedConfig):
    model_type = "v6_dynfw"
    def __init__(self, vocab_size=151936, D=128, n_layer=2, nh=16, mlp_mult=64,
                 fw_arch="v6", **kwargs):
        self.vocab_size = vocab_size
        self.D = D
        self.n_layer = n_layer
        self.nh = nh
        self.mlp_mult = mlp_mult
        self.fw_arch = fw_arch
        super().__init__(**kwargs)


class V6HFModel(PreTrainedModel):
    config_class = V6HFConfig
    _tied_weights_keys = ()

    @property
    def all_tied_weights_keys(self):
        return {}

    def __init__(self, config):
        super().__init__(config)
        if config.fw_arch == "v6":
            from dynfw.models.fused_fw_fw_cycle import BDHBlockFWCycleLM
            self.inner = BDHBlockFWCycleLM(D=config.D, nh=config.nh, vocab=config.vocab_size,
                                           n_layer=config.n_layer, steps=1, mlp_mult=config.mlp_mult, W=256)
        else:
            from dynfw.models.fused_fw_dla_cycle import BDHBlockDLACycleLM
            self.inner = BDHBlockDLACycleLM(D=config.D, nh=config.nh, vocab=config.vocab_size,
                                            n_layer=config.n_layer, steps=1, mlp_mult=config.mlp_mult, W=256, K=8)

    def forward(self, input_ids, attention_mask=None, labels=None, **kwargs):
        lg, _ = self.inner.forward(input_ids, targets=None)  # [B,T,V]
        from transformers.modeling_outputs import CausalLMOutputWithPast
        return CausalLMOutputWithPast(logits=lg)

    def np(self):
        return self.inner.np()


# 注册到 AutoModel + AutoConfig（新版 transformers: register 需 (model_type, config_class) 双参）
from transformers import AutoConfig
AutoConfig.register('v6_dynfw', V6HFConfig)
AutoModelForCausalLM.register(V6HFConfig, V6HFModel)
