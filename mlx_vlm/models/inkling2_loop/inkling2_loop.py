import mlx.core as mx

from ..inkling2.inkling2 import Model as Inkling2Model
from .config import ModelConfig
from .language import LanguageModel
from .vision import VisionModel

_GATES = {"attn": "q", "k": "k", "v": "v", "mlp": "mlp"}
_GATE_WEIGHTS = {
    "down_weight": "down_proj",
    "forget_weight": "forget_gate_proj",
    "gain_weight": "message_gate_proj",
}


class Model(Inkling2Model):
    def __init__(self, config: ModelConfig):
        super(Inkling2Model, self).__init__()
        self.config = config
        self.model_type = config.model_type
        config.vision_config.text_hidden_size = config.text_config.hidden_size
        config.vision_config.out_hidden_size = config.text_config.hidden_size
        self.language_model = LanguageModel(config.text_config)
        self.vision_tower = VisionModel(config.vision_config)

    @staticmethod
    def cast_predicate(path):
        # Keep the FP32 loop gates and message queries native, like the
        # attention-residual queries and routing tensors of Inkling 2.
        return Inkling2Model.cast_predicate(path) and not any(
            s in path for s in ("forget_gate_proj", "message_gate_proj")
        )

    def sanitize(self, weights):
        loop, rest = {}, {}
        for key, value in weights.items():
            if key in (
                "model.llm.middle_cycle_cross_pass_queries",
                "model.llm.middle_cycle_cross_pass_kv_queries",
            ):
                loop[key.rsplit(".", 1)[1]] = value
            elif key.startswith("model.llm.layers.") and ".controllers." in key:
                layer, sub = key[len("model.llm.layers.") :].split(".", 1)
                _, gate, weight = sub.split(".")
                loop[
                    f"language_model.model.layers.{layer}."
                    f"{_GATES[gate]}_message_gate.{_GATE_WEIGHTS[weight]}.weight"
                ] = value
            else:
                rest[key] = value
        out = super().sanitize(rest)
        if "middle_cycle_cross_pass_queries" in loop:
            qm = loop.pop("middle_cycle_cross_pass_queries")
            kv = loop.pop("middle_cycle_cross_pass_kv_queries")
            n = qm.shape[0] // 2
            # Checkpoint rows: [query_i, mlp_i] and [key_i, value_i] per loop layer.
            qm = qm.reshape(n, 2, -1)
            kv = kv.reshape(n, 2, -1)
            out["language_model.model.message_input.query"] = mx.stack(
                [qm[:, 0], kv[:, 0], kv[:, 1], qm[:, 1]], axis=1
            )
        out.update(loop)
        return out
