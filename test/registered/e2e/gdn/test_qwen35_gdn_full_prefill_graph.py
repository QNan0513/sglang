"""Full prefill CUDA graphs on the hybrid Qwen3.5 GDN architecture (#48593).

A full prefill graph runs the GDN extend on the static per-bucket tables of
linear/gdn_prefill_graph.py. Against a breakable prefill graph on the same
Triton GDN kernels, prompt logprobs and greedy tokens must match bit for bit:
fresh prompts on padded buckets, a multi-request batch, and a radix-cache
prefix hit that resumes from a checkpoint the full graph saved.
"""

import os
import random
import unittest

import torch

from sglang.srt.entrypoints.engine import Engine
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.test_utils import (
    DEFAULT_HYBRID_GDN_SMALL_MODEL_NAME_FOR_TEST,
    CustomTestCase,
)

register_cuda_ci(est_time=180, stage="extra-a", runner_config="1-gpu-small")

SAMPLING = {"max_new_tokens": 8, "temperature": 0}


def _prompts():
    rng = random.Random(0)

    def ids(n):
        return [rng.randrange(1000, 100000) for _ in range(n)]

    return {
        "singles": [ids(n) for n in (5, 37, 300, 2000)],
        "batch": [ids(n) for n in (100, 700, 1500)],
        "prefix": ids(1000),
        "tail": ids(64),
    }


def _result(out):
    meta = out["meta_info"]
    return {
        "input_logprobs": [lp for lp, _, _ in meta["input_token_logprobs"]],
        "output_ids": [token for _, token, _ in meta["output_token_logprobs"]],
        "cached_tokens": meta["cached_tokens"],
    }


class TestQwen35GDNFullPrefillGraph(CustomTestCase):
    model = os.environ.get(
        "QWEN35_GDN_TEST_MODEL", DEFAULT_HYBRID_GDN_SMALL_MODEL_NAME_FOR_TEST
    )

    def _run(self, cuda_graph_backend_prefill):
        prompts = _prompts()
        engine = Engine(
            model_path=self.model,
            trust_remote_code=True,
            attention_backend="flashinfer",
            linear_attn_prefill_backend="triton",
            cuda_graph_backend_prefill=cuda_graph_backend_prefill,
            mem_fraction_static=0.7,
            log_level="error",
        )
        try:

            def generate(input_ids, logprob_start_len=0):
                out = engine.generate(
                    input_ids=input_ids,
                    sampling_params=SAMPLING,
                    return_logprob=True,
                    logprob_start_len=logprob_start_len,
                )
                return [_result(o) for o in (out if isinstance(out, list) else [out])]

            results = {}
            for ids in prompts["singles"]:
                engine.flush_cache()
                results[f"single_{len(ids)}"] = generate(ids)
            engine.flush_cache()
            results["batch"] = generate(prompts["batch"])
            engine.flush_cache()
            generate(prompts["prefix"])
            # Logprobs only past the prefix, so the radix cache may serve it.
            results["prefix_hit"] = generate(
                prompts["prefix"] + prompts["tail"],
                logprob_start_len=len(prompts["prefix"]),
            )
            return results
        finally:
            engine.shutdown()
            torch.cuda.empty_cache()

    def test_full_prefill_graph_matches_breakable(self):
        full = self._run("full")
        breakable = self._run("breakable")
        self.assertGreater(full["prefix_hit"][0]["cached_tokens"], 0)
        for name, ref in breakable.items():
            for i, (got, want) in enumerate(zip(full[name], ref)):
                with self.subTest(request=name, row=i):
                    self.assertEqual(got["cached_tokens"], want["cached_tokens"])
                    self.assertEqual(got["input_logprobs"], want["input_logprobs"])
                    self.assertEqual(got["output_ids"], want["output_ids"])


if __name__ == "__main__":
    unittest.main()
