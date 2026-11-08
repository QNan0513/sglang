"""Full prefill CUDA graphs and the hybrid linear backend's extend metadata.

A full prefill graph has no eager break, so the hybrid backend runs the linear
extend on the linear child's static per-bucket tables: one set allocated per
captured bucket and refreshed in place before every replay, holding the real
requests only (the runner pads the request axis with zero-length sentinel
slots). The full-attention child still plans its own capture-stable metadata.
Decode-shaped metadata, and linear backends without such tables, keep the
per-child path.
"""

import unittest
from types import SimpleNamespace

import torch

from sglang.srt.layers.attention.hybrid_linear_attn_backend import (
    HybridLinearAttnBackend,
)
from sglang.srt.model_executor.forward_mode import ForwardMode
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


class _Child:
    token_to_kv_pool = None
    req_to_token_pool = None
    runner_ids = None
    needs_cpu_seq_lens = False
    use_captured_forward_metadata_for_breakable_cuda_graph = False

    def __init__(self, full_tables=False):
        self.full_tables = full_tables
        self.calls = []

    def init_forward_metadata_out_graph(self, forward_batch, in_capture=False):
        self.calls.append(("out_graph", forward_batch, in_capture))

    def init_prefill_cuda_graph_state(self, *, max_num_tokens, max_req_slots):
        self.calls.append(("prefill_state", max_num_tokens, max_req_slots))

    def supports_full_prefill_graph_extend(self):
        return self.full_tables

    def init_prefill_graph_metadata(self, forward_batch):
        tables = SimpleNamespace(bucket=forward_batch.input_ids.shape[0])
        self.calls.append(("init_tables", forward_batch, tables))
        return tables

    def refresh_prefill_graph_metadata(self, tables, forward_batch):
        self.calls.append(("refresh_tables", forward_batch, tables))


def _capture_batch(bucket, slots):
    """The runner's capture batch: one synthetic request, then sentinels."""
    lens = [bucket] + [0] * (slots - 1)
    return SimpleNamespace(
        forward_mode=ForwardMode.EXTEND,
        batch_size=slots,
        input_ids=torch.zeros(bucket, dtype=torch.int64),
        req_pool_indices=torch.arange(slots),
        seq_lens=torch.tensor(lens),
        extend_seq_lens=torch.tensor(lens),
        extend_prefix_lens=torch.zeros(slots, dtype=torch.int64),
        extend_start_loc=torch.tensor([0] + [bucket] * (slots - 1)),
        extend_seq_lens_cpu=lens,
        extend_prefix_lens_cpu=[0] * slots,
        mamba_track_mask=torch.zeros(slots, dtype=torch.bool),
        mamba_track_indices=torch.zeros(slots, dtype=torch.int64),
        mamba_track_seqlens=torch.zeros(slots, dtype=torch.int64),
        mamba_prefill_track_mask_cpu=None,
        mamba_track_seqlens_cpu=None,
    )


def _replay_view(bucket, slots, lens, prefixes):
    """The runner's slot-padded replay view: device per-request tensors padded
    to ``slots``; CPU lengths and track tensors are the real batch's."""
    bs, pad = len(lens), slots - len(lens)
    starts = [sum(lens[:i]) for i in range(bs)]
    return SimpleNamespace(
        forward_mode=ForwardMode.EXTEND,
        batch_size=slots,
        input_ids=torch.zeros(bucket, dtype=torch.int64),
        req_pool_indices=torch.tensor([7 + i for i in range(bs)] + [0] * pad),
        seq_lens=torch.tensor([n + p for n, p in zip(lens, prefixes)] + [0] * pad),
        extend_seq_lens=torch.tensor(lens + [0] * pad),
        extend_prefix_lens=torch.tensor(prefixes + [0] * pad),
        extend_start_loc=torch.tensor(starts + [sum(lens)] * pad),
        extend_seq_lens_cpu=list(lens),
        extend_prefix_lens_cpu=list(prefixes),
        mamba_track_mask=torch.tensor([True] + [False] * (bs - 1)),
        mamba_track_indices=torch.arange(bs) + 100,
        mamba_track_seqlens=torch.tensor(prefixes),
        mamba_prefill_track_mask_cpu=[True] + [False] * (bs - 1),
        mamba_track_seqlens_cpu=list(prefixes),
    )


def _out_graph_calls(child):
    """(batch identity, in_capture) of each out-of-graph metadata call."""
    return [
        (id(batch), flag) for kind, batch, flag in child.calls if kind == "out_graph"
    ]


def _backend(full_tables=True, slots=4):
    full, linear = _Child(), _Child(full_tables)
    backend = HybridLinearAttnBackend(full, linear, [0])
    backend.init_prefill_cuda_graph_state(max_num_tokens=64, max_req_slots=slots)
    full.calls.clear()
    linear.calls.clear()
    return backend, full, linear


class TestFullPrefillGraphLinearExtend(CustomTestCase):
    def test_prefill_graph_state_reaches_both_children(self):
        full, linear = _Child(), _Child(full_tables=True)
        backend = HybridLinearAttnBackend(full, linear, [0])
        backend.init_prefill_cuda_graph_state(max_num_tokens=64, max_req_slots=4)
        for child in (full, linear):
            self.assertEqual(child.calls, [("prefill_state", 64, 4)])

    def test_capture_and_replay_share_one_table_set_per_bucket(self):
        backend, full, linear = _backend()
        captures = {bucket: _capture_batch(bucket, 4) for bucket in (64, 16)}
        for batch in captures.values():
            backend.init_forward_metadata_out_graph(batch, in_capture=True)
        self.assertEqual(
            _out_graph_calls(full), [(id(b), True) for b in captures.values()]
        )
        tables = {}
        for (kind, batch, table), bucket in zip(linear.calls, captures):
            self.assertEqual(kind, "init_tables")
            self.assertEqual(table.bucket, bucket)
            self.assertEqual(batch.batch_size, 1)
            self.assertEqual(batch.extend_seq_lens_cpu, [bucket])
            self.assertEqual(batch.extend_prefix_lens_cpu, [0])
            self.assertEqual(batch.req_pool_indices.tolist(), [0])
            self.assertEqual(batch.mamba_track_indices.shape[0], 1)
            tables[bucket] = table

        full.calls.clear()
        linear.calls.clear()
        view = _replay_view(16, 4, lens=[5, 6, 3], prefixes=[0, 64, 9])
        backend.init_forward_metadata_out_graph(view)
        self.assertEqual(_out_graph_calls(full), [(id(view), False)])
        [(kind, batch, table)] = linear.calls
        self.assertEqual(kind, "refresh_tables")
        self.assertIs(table, tables[16])
        self.assertEqual(batch.batch_size, 3)
        self.assertEqual(batch.req_pool_indices.tolist(), [7, 8, 9])
        self.assertEqual(batch.extend_seq_lens.tolist(), [5, 6, 3])
        self.assertEqual(batch.extend_prefix_lens.tolist(), [0, 64, 9])
        self.assertEqual(batch.extend_start_loc.tolist(), [0, 5, 11])
        self.assertEqual(batch.mamba_prefill_track_mask_cpu, [True, False, False])
        # The runner's view itself is left as the full-attention child saw it.
        self.assertEqual(view.batch_size, 4)
        self.assertEqual(view.req_pool_indices.shape[0], 4)

    def test_a_replay_filling_every_slot_is_handed_over_as_is(self):
        backend, _, linear = _backend()
        backend.init_forward_metadata_out_graph(_capture_batch(16, 4), in_capture=True)
        linear.calls.clear()
        view = _replay_view(16, 4, lens=[4, 4, 4, 4], prefixes=[0, 0, 0, 0])
        backend.init_forward_metadata_out_graph(view)
        [(_, batch, _)] = linear.calls
        self.assertIs(batch, view)

    def test_decode_and_verify_keep_the_per_child_metadata(self):
        backend, full, linear = _backend()
        for mode in (ForwardMode.DECODE, ForwardMode.TARGET_VERIFY):
            batch = SimpleNamespace(forward_mode=mode)
            with self.subTest(mode=mode):
                full.calls.clear()
                linear.calls.clear()
                backend.init_forward_metadata_out_graph(batch, in_capture=True)
                for child in (full, linear):
                    self.assertEqual(_out_graph_calls(child), [(id(batch), True)])
                    self.assertEqual(len(child.calls), 1)

    def test_linear_backends_without_full_tables_keep_the_per_child_metadata(self):
        backend, full, linear = _backend(full_tables=False)
        batch = _capture_batch(16, 4)
        backend.init_forward_metadata_out_graph(batch, in_capture=True)
        for child in (full, linear):
            self.assertEqual(_out_graph_calls(child), [(id(batch), True)])
            self.assertEqual(len(child.calls), 1)


if __name__ == "__main__":
    unittest.main()
