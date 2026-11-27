"""Transformers residual-stream steering with scoped decoder-layer hooks."""

from __future__ import annotations

from contextlib import AbstractContextManager, nullcontext
from dataclasses import dataclass
from typing import Any, Literal


PositionPolicy = Literal["last", "all"]


@dataclass(frozen=True)
class TransformersConfig:
    model: str
    max_new_tokens: int = 128
    temperature: float = 0.0
    top_p: float = 1.0
    batch_size: int = 1
    dtype: str = "bfloat16"
    trust_remote_code: bool = True
    attention_implementation: str = "eager"
    device_map: str | None = None
    max_model_len: int | None = None
    rope_scaling: dict[str, Any] | None = None


@dataclass(frozen=True)
class SteeringSpec:
    """One vector per decoder layer, scaled as a residual-norm fraction."""

    vectors: dict[int, Any]
    residual_norms: dict[int, float]
    alpha: float
    position_policy: PositionPolicy = "last"

    @property
    def layers(self) -> tuple[int, ...]:
        return tuple(sorted(self.vectors))


def decoder_layers(model: Any) -> Any:
    """Return the decoder-layer sequence for common HF causal-LM wrappers."""

    candidates = (
        getattr(getattr(model, "model", None), "layers", None),
        getattr(
            getattr(getattr(model, "model", None), "model", None),
            "layers",
            None,
        ),
    )
    for layers in candidates:
        if layers is not None:
            return layers
    raise ValueError("Could not locate decoder layers on the Transformers model")


def add_steering_to_output(
    output: Any,
    delta: Any,
    position_policy: PositionPolicy = "last",
) -> Any:
    """Add a steering delta while preserving tensor or tuple output structure."""

    hidden = output[0] if isinstance(output, tuple) else output
    vector = delta.to(device=hidden.device, dtype=hidden.dtype)
    adjusted = hidden.clone()
    if position_policy == "last":
        adjusted[:, -1, :] = adjusted[:, -1, :] + vector
    elif position_policy == "all":
        adjusted = adjusted + vector.view(1, 1, -1)
    else:
        raise ValueError(f"Unknown position policy {position_policy!r}")
    if isinstance(output, tuple):
        return (adjusted, *output[1:])
    return adjusted


class SteeringController(AbstractContextManager):
    """Install steering hooks for one generation call and always remove them."""

    def __init__(self, model: Any, spec: SteeringSpec):
        self.model = model
        self.spec = spec
        self.handles: list[Any] = []

    def __enter__(self):
        import torch

        layers = decoder_layers(self.model)
        for layer_index in self.spec.layers:
            if layer_index < 0 or layer_index >= len(layers):
                raise ValueError(
                    f"Layer {layer_index} is outside [0, {len(layers) - 1}]"
                )
            unit = torch.as_tensor(self.spec.vectors[layer_index], dtype=torch.float32)
            unit_norm = float(torch.linalg.vector_norm(unit))
            if not 0.999 <= unit_norm <= 1.001:
                raise ValueError(
                    f"Steering vector at layer {layer_index} is not unit norm: "
                    f"{unit_norm:.6f}"
                )
            scale = self.spec.alpha * self.spec.residual_norms[layer_index]
            delta = unit * scale

            def hook(_module, _inputs, output, *, layer_delta=delta):
                return add_steering_to_output(
                    output,
                    layer_delta,
                    position_policy=self.spec.position_policy,
                )

            self.handles.append(layers[layer_index].register_forward_hook(hook))
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        for handle in self.handles:
            handle.remove()
        self.handles.clear()
        return False


class SteeredGenerator:
    """A deterministic Transformers generator with optional scoped steering."""

    def __init__(
        self,
        config: TransformersConfig,
        *,
        model: Any | None = None,
        tokenizer: Any | None = None,
    ):
        import torch
        from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

        self.config = config
        self.torch = torch
        self.tokenizer = tokenizer or AutoTokenizer.from_pretrained(
            config.model,
            trust_remote_code=config.trust_remote_code,
        )
        if model is None:
            load_kwargs: dict[str, Any] = {
                "dtype": getattr(torch, config.dtype),
                "trust_remote_code": config.trust_remote_code,
                "attn_implementation": config.attention_implementation,
            }
            if config.rope_scaling is not None:
                hf_config = AutoConfig.from_pretrained(
                    config.model,
                    trust_remote_code=config.trust_remote_code,
                )
                rope_scaling = dict(config.rope_scaling)
                hf_config.rope_scaling = rope_scaling
                hf_config.rope_parameters = dict(rope_scaling)
                load_kwargs["config"] = hf_config
            if config.device_map is not None:
                load_kwargs["device_map"] = config.device_map
            model = AutoModelForCausalLM.from_pretrained(
                config.model,
                **load_kwargs,
            )
            if config.device_map is None and torch.cuda.is_available():
                model = model.to("cuda")
        self.model = model.eval()
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.padding_side = "left"

    @property
    def input_device(self):
        return self.model.get_input_embeddings().weight.device

    def render_chat(self, prompts: list[str], disable_thinking: bool = True) -> list[str]:
        rendered = []
        for prompt in prompts:
            content = f"{prompt}\n/no_think" if disable_thinking else prompt
            messages = [{"role": "user", "content": content}]
            kwargs = {
                "tokenize": False,
                "add_generation_prompt": True,
            }
            try:
                text = self.tokenizer.apply_chat_template(
                    messages,
                    enable_thinking=not disable_thinking,
                    **kwargs,
                )
            except TypeError:
                text = self.tokenizer.apply_chat_template(messages, **kwargs)
            rendered.append(text)
        return rendered

    def generate(
        self,
        prompts: list[str],
        steering: SteeringSpec | None = None,
    ) -> list[str]:
        results: list[str] = []
        context = (
            nullcontext()
            if steering is None or steering.alpha == 0
            else SteeringController(self.model, steering)
        )
        with context:
            for start in range(0, len(prompts), self.config.batch_size):
                batch = prompts[start : start + self.config.batch_size]
                inputs = self.tokenizer(
                    batch,
                    return_tensors="pt",
                    padding=True,
                )
                inputs = {
                    key: value.to(self.input_device)
                    for key, value in inputs.items()
                }
                with self.torch.inference_mode():
                    output_ids = self.model.generate(
                        **inputs,
                        do_sample=self.config.temperature > 0,
                        temperature=(
                            None
                            if self.config.temperature == 0
                            else self.config.temperature
                        ),
                        top_p=self.config.top_p,
                        max_new_tokens=self.config.max_new_tokens,
                        pad_token_id=self.tokenizer.pad_token_id,
                        eos_token_id=self.tokenizer.eos_token_id,
                        use_cache=True,
                    )
                prompt_width = inputs["input_ids"].shape[1]
                for ids in output_ids:
                    results.append(
                        self.tokenizer.decode(
                            ids[prompt_width:],
                            skip_special_tokens=True,
                        )
                    )
        return results
