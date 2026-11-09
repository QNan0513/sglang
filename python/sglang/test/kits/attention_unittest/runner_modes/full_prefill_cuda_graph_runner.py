"""Plain EXTEND under a full prefill CUDA graph, as PrefillCudaGraphRunner's
full backend drives it: one graph per token bucket, largest first, each captured
on a single synthetic request followed by zero-length sentinel slots. A real
batch replays the smallest bucket that holds its tokens, padded to the same
request slots, after the backend plans it out of graph. Each replay must match
the reference and the eager forward of the same batch.
"""

import torch

from sglang.srt.model_executor.forward_context import ForwardContext, forward_context
from sglang.srt.model_executor.forward_mode import ForwardMode

from ..attention_methods.dense_attention import (
    DEFAULT_DEVICE,
    DEFAULT_DTYPE,
    DEFAULT_HEAD_DIM,
    DEFAULT_HIDDEN_SIZE,
    DEFAULT_MAX_CONTEXT_LEN,
    DENSE_ATOL,
    DENSE_RTOL,
    DenseAttentionCase,
    _make_forward_batch,
    build_dense_attention_fixture,
    expected_dense_output_from_inputs,
    make_dense_random_inputs,
    prepare_dense_runner_inputs,
)


def _slot_padded_case(
    like: DenseAttentionCase,
    name: str,
    *,
    req_slots: int,
    prefix_lens: tuple[int, ...],
    extend_lens: tuple[int, ...],
) -> DenseAttentionCase:
    pad = req_slots - len(prefix_lens)
    return DenseAttentionCase(
        name=name,
        backend=like.backend,
        forward_mode=ForwardMode.EXTEND,
        num_heads=like.num_heads,
        num_kv_heads=like.num_kv_heads,
        page_size=like.page_size,
        prefix_lens=prefix_lens + (0,) * pad,
        extend_lens=extend_lens + (0,) * pad,
    )


def run_dense_full_prefill_cuda_graph_cases(
    testcase,
    cases: tuple[DenseAttentionCase, ...],
    *,
    buckets: tuple[int, ...],
    req_slots: int,
    head_dim: int = DEFAULT_HEAD_DIM,
    hidden_size: int = DEFAULT_HIDDEN_SIZE,
    max_context_len: int = DEFAULT_MAX_CONTEXT_LEN,
    dtype: torch.dtype = DEFAULT_DTYPE,
    device: str = DEFAULT_DEVICE,
):
    """Capture every bucket, then replay each EXTEND case (at most
    ``req_slots`` requests) through the smallest bucket that holds it.
    Returns the backend under test."""
    buckets = tuple(sorted(buckets, reverse=True))
    like = cases[0]
    capture_cases = [
        _slot_padded_case(
            like,
            f"{like.name}_full_prefill_capture_{bucket}",
            req_slots=req_slots,
            prefix_lens=(0,),
            extend_lens=(bucket,),
        )
        for bucket in buckets
    ]
    # The contiguous layout is the one _make_forward_batch and
    # prepare_dense_runner_inputs write for the replays.
    fixture = build_dense_attention_fixture(
        testcase,
        capture_cases[0],
        head_dim=head_dim,
        hidden_size=hidden_size,
        max_context_len=max_context_len,
        dtype=dtype,
        device=device,
        runner_batch_size=req_slots,
        loc_layout="contiguous",
    )
    backend, module, runner = fixture.backend, fixture.actual_module, fixture.runner
    static_hidden = torch.zeros(buckets[0], hidden_size, dtype=dtype, device=device)

    caller_stream = torch.cuda.current_stream(device=device)
    stream = torch.cuda.Stream(device=device)
    stream.wait_stream(caller_stream)
    try:
        with (
            torch.no_grad(),
            torch.cuda.stream(stream),
            forward_context(ForwardContext(attn_backend=backend)),
        ):
            backend.init_prefill_cuda_graph_state(
                max_num_tokens=buckets[0], max_req_slots=req_slots
            )
            graphs = {}
            for bucket, capture_case in zip(buckets, capture_cases):
                capture_batch = _make_forward_batch(
                    capture_case, runner, max_context_len=max_context_len, device=device
                )
                hidden = static_hidden[:bucket]
                backend.init_forward_metadata_out_graph(capture_batch, in_capture=True)
                module(hidden, capture_batch)  # warmup, as the runner does
                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph, stream=stream):
                    output = module(hidden, capture_batch)
                graphs[bucket] = (graph, capture_batch, output)

            for case in cases:
                if case.batch_size > req_slots or case.num_input_tokens > buckets[0]:
                    raise ValueError(f"{case.name} does not fit the captured graphs")
                bucket = min(b for b in buckets if b >= case.num_input_tokens)
                graph, capture_batch, output = graphs[bucket]
                num_tokens = case.num_input_tokens
                batch = _make_forward_batch(
                    case, runner, max_context_len=max_context_len, device=device
                )
                padded_batch = _make_forward_batch(
                    _slot_padded_case(
                        case,
                        f"{case.name}_full_prefill_replay",
                        req_slots=req_slots,
                        prefix_lens=case.prefix_lens,
                        extend_lens=case.extend_lens,
                    ),
                    runner,
                    max_context_len=max_context_len,
                    device=device,
                )
                inputs = make_dense_random_inputs(
                    case, fixture, dtype=dtype, device=device
                )
                # The captures wrote KV over request 0's slots.
                prepare_dense_runner_inputs(
                    fixture, case, batch, inputs, max_context_len=max_context_len
                )
                expected = expected_dense_output_from_inputs(
                    fixture, case, inputs, None
                )
                backend.init_forward_metadata(batch)
                eager = module(inputs["input_hidden"], batch)

                # The graph reads its inputs and write slots by address; tokens
                # past the batch write to slot 0.
                static_hidden[:num_tokens].copy_(inputs["input_hidden"])
                static_hidden[num_tokens:bucket].normal_()
                capture_batch.out_cache_loc[:num_tokens].copy_(
                    padded_batch.out_cache_loc
                )
                capture_batch.out_cache_loc[num_tokens:].zero_()
                backend.init_forward_metadata_out_graph(padded_batch)
                graph.replay()
                actual = output[:num_tokens].clone()

                torch.testing.assert_close(
                    actual, expected, atol=DENSE_ATOL, rtol=DENSE_RTOL
                )
                torch.testing.assert_close(
                    actual, eager, atol=DENSE_ATOL, rtol=DENSE_RTOL
                )
    finally:
        caller_stream.wait_stream(stream)
    return backend
