"""
Minimal Blend Inference Script (attn only)

Runs one attn-blend inference on one synthetic example.

Usage:
    python sgblend_blend.py --model_path /path/to/model
"""

import argparse
import time
from pathlib import Path
from typing import Tuple, List, Dict

import sglang as sgl
from sglang.srt.utils.triton_attention_score import warmup_triton_kernels


# ==================== Configuration ====================
BLEND_SEP = " <|blendsep|>"

# Model template configurations
TEMPLATES = {
    "llama": (
        "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n",
        "<|eot_id|><|start_header_id|>user<|end_header_id|>\n\n",
        "<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n",
    ),
    "mistral": ("<s>[INST]", "", "[/INST]"),
    "qwen": (
        "<|im_start|>system\n",
        "<|im_end|>\n<|im_start|>user\n",
        "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n",
    ),
}

class LongBenchBlendEngine:
    """LongBench Blend Inference Engine (attn only)."""

    def __init__(
        self,
        model_path: str,
        gpu_id: int,
        context_enhance: bool = False,
    ):
        self.context_enhance = context_enhance
        self.model_name = Path(model_path).name.lower()
        self.model_path = model_path
        self.context_length = 10000
        self.first_style = "KVCOMPUTE"
        self.start = 0
        self.method = "attn"

        self.llm = sgl.Engine(
            model_path=model_path,
            mem_fraction_static=0.8,
            context_length=self.context_length,
            tp_size=1,
            disable_cuda_graph=True,
            trust_remote_code=True,
            disable_radix_cache=True,
            chunked_prefill_size=-1,
            dtype="bfloat16",
            attention_backend="triton"
        )

        # Load tokenizer for length checking
        from transformers import AutoTokenizer

        self.tokenizer = AutoTokenizer.from_pretrained(
            model_path, trust_remote_code=True
        )

        self.attn_start = 0
        self.attn_end = -1
        self._model_config = None

    def _get_model_config(self) -> dict:
        """Get model architecture parameters (cached)."""
        if self._model_config is not None:
            return self._model_config

        from transformers import AutoConfig

        config = AutoConfig.from_pretrained(self.model_path, trust_remote_code=True)

        head_dim = getattr(config, "head_dim", None)
        if head_dim is None:
            head_dim = config.hidden_size // config.num_attention_heads

        self._model_config = {
            "head_dim": head_dim,
            "num_layers": getattr(config, "num_hidden_layers", 32),
            "num_heads": getattr(config, "num_attention_heads", 32),
            "num_kv_heads": getattr(
                config, "num_key_value_heads", config.num_attention_heads
            ),
        }
        return self._model_config

    def warmup(self, num_warmup: int = 3):
        """Warmup inference to eliminate cold start effects."""
        print(f"[Warmup] Starting {num_warmup} warmup iterations...")

        cfg = self._get_model_config()
        print(
            f"[Warmup] Model: head_dim={cfg['head_dim']}, layers={cfg['num_layers']}, "
            f"heads={cfg['num_heads']}, kv_heads={cfg['num_kv_heads']}"
        )

        warmup_triton_kernels(
            head_dims=[cfg["head_dim"]],
            num_warmup_iters=3,
            num_layers=cfg["num_layers"],
            num_heads=cfg["num_heads"],
            num_kv_heads=cfg["num_kv_heads"],
        )

        sys_h, sys_e, asst_h = self._get_template()
        warmup_prompt = (
            sys_h
            + "You are a helpful assistant."
            + sys_e
            + "Hello, how are you?"
            + asst_h
        )

        for i in range(num_warmup):
            for _ in self.llm.generate(
                warmup_prompt,
                {"temperature": 0, "max_new_tokens": 1},
                stream=True,
                blend_style=None,
            ):
                pass
            print(f"[Warmup] Iteration {i + 1}/{num_warmup} done")

        print("[Warmup] Completed.")

    def warmup_blend(self, ratio: float):
        """Warmup attn blend path with a tiny synthetic prompt."""
        prefix = "You are a helpful assistant.\n\n## Passages\n"
        docs = [
            "Warmup passage." * 100,
            "Warmup passage." * 100,
            "Warmup passage." * 100,
        ]
        q_prompt = ["\n## Question\n", "Warmup question?" * 10, "\n"]
        for _ in range(3):
            self.run(prefix, docs, q_prompt, ratio=ratio, max_tokens=1)

    def _get_template(self) -> Tuple[str, str, str]:
        """Get model template based on model name."""
        for prefix, template in TEMPLATES.items():
            if self.model_name.startswith(prefix):
                return template
        return ("", "", "")

    def _build_prompt(
        self, system_prompt: str, docs: List[str], q_prompt: List[str], use_sep: bool
    ) -> Tuple[str, str]:
        """Build complete prompt from components."""
        sys_h, sys_e, asst_h = self._get_template()
        prefix = sys_h + system_prompt + sys_e
        suffix = "".join(q_prompt) + "\n\n## Answer\n" + asst_h

        if use_sep:
            query_sep = BLEND_SEP.join(q_prompt)
            return BLEND_SEP.join([prefix] + docs + [suffix]), query_sep
        return prefix + "".join(docs) + suffix, suffix

    def check_prompt_length(
        self,
        system_prompt: str,
        docs: List[str],
        q_prompt: List[str],
        max_new_tokens: int,
    ) -> Tuple[bool, int]:
        """Check if prompt length exceeds context_length.

        Returns:
            (is_valid, token_count): is_valid=True if within limit
        """
        prompt, _ = self._build_prompt(system_prompt, docs, q_prompt, use_sep=False)
        token_count = len(self.tokenizer.encode(prompt))
        max_allowed = self.context_length - max_new_tokens
        return token_count <= max_allowed, token_count

    def run(
        self,
        system_prompt: str,
        docs: List[str],
        q_prompt: List[str],
        ratio: float,
        max_tokens: int,
    ) -> Dict[str, object]:
        """Execute one attn-blend inference and return timing/text."""
        params = {"temperature": 0, "max_new_tokens": max_tokens}

        prompt, query_sep = self._build_prompt(
            system_prompt, docs, q_prompt, use_sep=True
        )
        blend_args = {
            "blend_style": self.first_style,
            "separator": BLEND_SEP,
            "start": self.start,
            "ratio": ratio,
            "method": self.method,
        }
        if self.context_enhance:
            blend_args["is_contextblend"] = True

        # Phase 1: KV collection
        for _ in self.llm.generate(
            prompt, {"temperature": 0, "max_new_tokens": 1}, stream=True, **blend_args
        ):
            pass

        # Phase 2: Q collection
        q_time = 0.0
        start_q = time.time()
        blend_args["blend_style"] = "QCOMPUTE"
        blend_args["attn_start"] = self.attn_start
        blend_args["attn_end"] = self.attn_end
        for _ in self.llm.generate(
            query_sep,
            {"temperature": 0, "max_new_tokens": 1},
            stream=True,
            **blend_args,
        ):
            pass
        q_time = time.time() - start_q

        blend_args["blend_style"] = "DO_BLEND_FINISH"
        start_time = time.time()
        ttft, text = None, ""
        for out in self.llm.generate(prompt, params, stream=True, **blend_args):
            if ttft is None and out.get("text"):
                ttft = time.time() - start_time
            text = out.get("text", "")

        ttft = ttft or (time.time() - start_time)
        return {
            "text": text,
            "ttft": ttft + q_time,
            "decode_time": time.time() - start_time - ttft,
        }

    def shutdown(self):
        if hasattr(self, "llm"):
            self.llm.shutdown()


def main():
    parser = argparse.ArgumentParser(description="Minimal attn-blend single synthetic run")
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--ratio", type=float, default=0.3)
    parser.add_argument("--max_new_tokens", type=int, default=32)
    parser.add_argument(
        "--context_enhance",
        action="store_true",
        default=False,
        help="Enable context-enhanced QCOMPUTE (inject compressed context KV into query forward)",
    )
    parser.add_argument(
        "--attn_layer",
        type=str,
        default="all_layer",
        choices=["best_layer", "last_layer", "all_layer"],
        help="Which layers to use for attn-based Q collection: best_layer (model-specific), last_layer, all_layer",
    )
    args = parser.parse_args()

    print(f"[Init] model={args.model_path}")
    print(f"[Init] ratio={args.ratio}, context_enhance={args.context_enhance}")

    engine = LongBenchBlendEngine(
        args.model_path, args.gpu, context_enhance=args.context_enhance
    )

    try:
        num_layers = engine._get_model_config()["num_layers"]
        model_name = Path(args.model_path).name.lower()
        if args.attn_layer == "best_layer":
            if model_name.startswith("qwen3-8b"):
                engine.attn_start, engine.attn_end = 20, 21
            elif model_name.startswith("llama"):
                engine.attn_start, engine.attn_end = 13, 14
            elif model_name.startswith("mistral"):
                engine.attn_start, engine.attn_end = 16, 17
            else:
                engine.attn_start, engine.attn_end = num_layers - 1, num_layers
        elif args.attn_layer == "last_layer":
            engine.attn_start, engine.attn_end = num_layers - 1, num_layers
        else:
            engine.attn_start, engine.attn_end = 0, num_layers

        # One synthetic example (no dataset loading).
        system_prompt = "You are a helpful assistant."
        docs = [
            "Passage:\nParis is the capital and most populous city of France.\n\n"*100,
            "Passage:\nFrance is a country in Europe.\n\n"*100,
            "Passage:\nThe Eiffel Tower is located in Paris.\n\n"*100,
        ]
        q_prompt = [
            "\n## Question\n",
            "What is the capital of France?",
        ]
        max_tokens = args.max_new_tokens

        is_valid, token_count = engine.check_prompt_length(
            system_prompt, docs, q_prompt, max_tokens
        )
        if not is_valid:
            raise ValueError(
                f"Prompt too long: {token_count} > {engine.context_length - max_tokens}"
            )

        print("[Run] warmup...")
        engine.warmup(num_warmup=1)
        engine.warmup_blend(ratio=args.ratio)

        print("[Run] inference...")
        result = engine.run(
            system_prompt,
            docs,
            q_prompt,
            ratio=args.ratio,
            max_tokens=max_tokens,
        )

        print("=" * 60)
        print(result["text"])
    finally:
        engine.shutdown()


if __name__ == "__main__":
    main()
