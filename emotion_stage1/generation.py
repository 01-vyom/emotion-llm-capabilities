"""Generation backends for Stage 1.

Use vLLM on the A100 for real runs. The Transformers backend is slower but
useful for smoke tests and environments without vLLM installed.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any, Protocol


@dataclass(frozen=True)
class GenerationConfig:
    model: str
    max_new_tokens: int = 32
    temperature: float = 0.0
    top_p: float = 1.0
    batch_size: int = 32
    dtype: str = "bfloat16"
    trust_remote_code: bool = True
    max_model_len: int | None = None
    gpu_memory_utilization: float = 0.9
    enforce_eager: bool = False
    tensor_parallel_size: int = 1
    rope_scaling: dict[str, Any] | None = None


class Generator(Protocol):
    def generate(self, prompts: list[str]) -> list[str]:
        ...


def build_vllm_engine_kwargs(config: GenerationConfig) -> dict[str, Any]:
    kwargs: dict[str, Any] = {
        "model": config.model,
        "dtype": config.dtype,
        "trust_remote_code": config.trust_remote_code,
        "gpu_memory_utilization": config.gpu_memory_utilization,
        "enforce_eager": config.enforce_eager,
        "tensor_parallel_size": config.tensor_parallel_size,
    }
    if config.max_model_len is not None:
        kwargs["max_model_len"] = config.max_model_len
    if config.rope_scaling is not None:
        rope_scaling = dict(config.rope_scaling)
        kwargs["hf_overrides"] = {
            # Transformers exposes rope_scaling; vLLM 0.19 Qwen3 reads
            # rope_parameters. Set both to support either config surface.
            "rope_scaling": rope_scaling,
            "rope_parameters": dict(rope_scaling),
        }
    return kwargs


class VLLMGenerator:
    def __init__(self, config: GenerationConfig):
        if config.rope_scaling is not None:
            os.environ.setdefault("VLLM_ALLOW_LONG_MAX_MODEL_LEN", "1")

        from vllm import LLM, SamplingParams

        self.config = config
        self.sampling_params = SamplingParams(
            temperature=config.temperature,
            top_p=config.top_p,
            max_tokens=config.max_new_tokens,
        )
        self.llm = LLM(**build_vllm_engine_kwargs(config))

    def generate(self, prompts: list[str]) -> list[str]:
        outputs = self.llm.generate(prompts, self.sampling_params)
        return [output.outputs[0].text for output in outputs]


class TransformersGenerator:
    def __init__(self, config: GenerationConfig):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.config = config
        self.torch = torch
        self.tokenizer = AutoTokenizer.from_pretrained(
            config.model,
            trust_remote_code=config.trust_remote_code,
        )
        self.model = AutoModelForCausalLM.from_pretrained(
            config.model,
            torch_dtype=getattr(torch, config.dtype),
            device_map="auto",
            trust_remote_code=config.trust_remote_code,
        )
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.padding_side = "left"

    def generate(self, prompts: list[str]) -> list[str]:
        results: list[str] = []
        for start in range(0, len(prompts), self.config.batch_size):
            batch = prompts[start : start + self.config.batch_size]
            inputs = self.tokenizer(batch, return_tensors="pt", padding=True)
            inputs = {key: value.to(self.model.device) for key, value in inputs.items()}
            with self.torch.inference_mode():
                output_ids = self.model.generate(
                    **inputs,
                    do_sample=self.config.temperature > 0,
                    temperature=None if self.config.temperature == 0 else self.config.temperature,
                    top_p=self.config.top_p,
                    max_new_tokens=self.config.max_new_tokens,
                    pad_token_id=self.tokenizer.pad_token_id,
                    eos_token_id=self.tokenizer.eos_token_id,
                )
            prompt_width = inputs["input_ids"].shape[1]
            for ids in output_ids:
                gen_ids = ids[prompt_width:]
                results.append(self.tokenizer.decode(gen_ids, skip_special_tokens=True))
        return results


def make_generator(engine: str, config: GenerationConfig) -> Generator:
    if engine == "vllm":
        return VLLMGenerator(config)
    if engine == "transformers":
        return TransformersGenerator(config)
    raise ValueError(f"Unknown engine {engine!r}; expected 'vllm' or 'transformers'")


def render_chat_prompts(
    prompts: list[str],
    model: str,
    disable_thinking: bool = True,
    trust_remote_code: bool = True,
) -> list[str]:
    """Render raw prompts through the model chat template.

    Qwen3 tokenizers support `enable_thinking=False`; for tokenizers without that
    keyword we fall back to a normal chat template. The raw prompt is also
    suffixed with `/no_think` as an extra Qwen-compatible guard when requested.
    """

    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model, trust_remote_code=trust_remote_code)
    rendered = []
    for prompt in prompts:
        content = prompt
        if disable_thinking:
            content = f"{content}\n/no_think"
        messages = [{"role": "user", "content": content}]
        try:
            text = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=not disable_thinking,
            )
        except TypeError:
            text = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
        rendered.append(text)
    return rendered
