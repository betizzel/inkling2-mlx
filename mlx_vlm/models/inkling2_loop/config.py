from dataclasses import dataclass
from typing import Optional

from ..base import BaseModelConfig
from ..inkling2.config import TextConfig as Inkling2TextConfig
from ..inkling2.config import VisionConfig, validate_sharding


@dataclass
class TextConfig(Inkling2TextConfig):
    model_type: str = "inkling2_loop_text"
    hidden_size: int = 2304
    num_hidden_layers: int = 24
    num_attention_heads: int = 24
    swa_num_attention_heads: int = 24
    log_scaling_n_floor: Optional[int] = 8192
    intermediate_size: int = 1152
    n_routed_experts: int = 128
    loop_config: Optional[dict] = None
    loop_layer_index_start: int = 6
    loop_layer_index_end: int = 18
    loop_num_passes: int = 3
    loop_gate_rank: int = 64

    def __post_init__(self):
        loop = self.loop_config or {}
        self.loop_layer_index_start = loop.get("core_start", self.loop_layer_index_start)
        self.loop_layer_index_end = loop.get("core_end", self.loop_layer_index_end)
        self.loop_num_passes = loop.get("num_passes", self.loop_num_passes)
        self.loop_gate_rank = loop.get("gate_rank", self.loop_gate_rank)
        if loop.get("is_varloop"):
            raise ValueError("Variable-depth Inkling 2 loops are not supported")
        block = self.n_layers_per_attn_res_block
        start, end = self.loop_layer_index_start, self.loop_layer_index_end
        if (
            self.loop_num_passes < 1
            or not 0 < start < end <= self.num_hidden_layers
            or start % block
            or end % block
        ):
            raise ValueError("Invalid Inkling 2 loop_config")
        self.loop_config = {
            "core_start": start,
            "core_end": end,
            "num_passes": self.loop_num_passes,
            "gate_rank": self.loop_gate_rank,
            "is_varloop": False,
        }

    @property
    def num_loop_layers(self):
        return self.loop_layer_index_end - self.loop_layer_index_start

    @property
    def total_num_hidden_layers(self):
        """Decoder layer applications per token, counting every loop pass."""
        return self.num_hidden_layers + (self.loop_num_passes - 1) * self.num_loop_layers


@dataclass
class ModelConfig(BaseModelConfig):
    model_type: str = "inkling2_loop"
    text_config: Optional[TextConfig] = None
    vision_config: Optional[VisionConfig] = None
    mtp_config: Optional[dict] = None
    image_token_id: int = 200054
    eos_token_id: int = 200006
    quantization: Optional[dict] = None
    quantization_config: Optional[dict] = None

    def __post_init__(self):
        if not isinstance(self.text_config, TextConfig):
            self.text_config = TextConfig.from_dict(self.text_config or {})
        if not isinstance(self.vision_config, VisionConfig):
            self.vision_config = VisionConfig.from_dict(self.vision_config or {})
        self.vision_config.text_hidden_size = self.text_config.hidden_size
        self.vision_config.out_hidden_size = self.text_config.hidden_size


__all__ = ["ModelConfig", "TextConfig", "VisionConfig", "validate_sharding"]
