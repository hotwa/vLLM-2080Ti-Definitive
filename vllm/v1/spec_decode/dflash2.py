# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""DFlash2 proposer for the V1 model-runner path.

Upstream PR #52816 runs DFlash2 through the V2 speculator. This fork only has
the V1 proposer path for DFlash, so DFlash2 is adapted here on top of
DFlashProposer: the draft model forward is unchanged (single parallel pass),
but draft tokens come from the DFlash2 candidate selector instead of an
argmax over draft lm_head logits.

The path walk is greedy (argmax at every step, lowest index wins ties), which
matches the fork's always-greedy draft policy and the upstream temperature==0
branch of the selector walk kernel. Rejection sampling on the target model
keeps decoding lossless.
"""

import torch
from typing_extensions import override

from vllm.logger import init_logger
from vllm.v1.attention.backend import CommonAttentionMetadata
from vllm.v1.spec_decode.dflash import DFlashProposer

logger = init_logger(__name__)


class DFlash2Proposer(DFlashProposer):
    def __init__(
        self,
        vllm_config,
        device: torch.device,
        runner=None,
    ):
        spec_config = vllm_config.speculative_config
        assert spec_config is not None and spec_config.method == "dflash"
        architectures = (
            getattr(spec_config.draft_model_config, "architectures", None) or []
        )
        if "DFlash2DraftModel" not in architectures:
            raise ValueError(
                "DFlash2Proposer requires a DFlash2 checkpoint "
                f"(architectures={architectures}); refusing to run the DFlash2 "
                "selector on a DFlash v1 draft model."
            )
        super().__init__(vllm_config, device, runner)

        draft_config = self.draft_model_config.hf_config.dflash_config
        self.selector_top_k = int(draft_config["selector_top_k"])
        self._dflash2_anchor_ids: torch.Tensor | None = None
        self._dflash2_num_reqs = 0

    @override
    def load_model(self, target_model) -> None:
        super().load_model(target_model)
        self._log_diagnostics()

    def _log_diagnostics(self) -> None:
        from vllm.distributed import get_tensor_model_parallel_world_size
        from vllm.model_executor.models.qwen3_dflash2 import _flashinfer_topk
        from vllm.platforms import current_platform

        spec_config = self.vllm_config.speculative_config
        compilation_config = getattr(self.vllm_config, "compilation_config", None)
        try:
            cc = current_platform.get_device_capability()
            sm = f"SM{cc.major}{cc.minor}" if cc is not None else "unknown"
        except Exception:
            sm = "unknown"
        draft_attn_backend = "unknown"
        for name, module in self.model.named_modules():
            if name in self._draft_attn_layer_names:
                impl = getattr(module, "impl", None)
                if impl is not None:
                    draft_attn_backend = type(impl).__name__
                break
        logger.info(
            "DFlash2 speculative decoding initialized:\n"
            "  target model:            %s\n"
            "  drafter model:           %s\n"
            "  DFlash version:          2\n"
            "  target TP size:          %d\n"
            "  draft TP size:           %s\n"
            "  speculative tokens:      %d\n"
            "  selector top-k:          %d (backend: %s)\n"
            "  drafter attention:       %s (non-causal)\n"
            "  CUDA graphs:             %s\n"
            "  SM capability:           %s\n"
            "  model runner:            V1 (DFlash2Proposer)",
            self.vllm_config.model_config.model,
            spec_config.model,
            get_tensor_model_parallel_world_size(),
            spec_config.draft_tensor_parallel_size,
            spec_config.num_speculative_tokens,
            self.selector_top_k,
            "flashinfer radix top-k"
            if _flashinfer_topk() is not None
            else "torch.topk",
            draft_attn_backend,
            getattr(compilation_config, "cudagraph_mode", "n/a"),
            sm,
        )

    @override
    def set_inputs_first_pass(
        self,
        target_token_ids: torch.Tensor,
        next_token_ids: torch.Tensor,
        target_positions: torch.Tensor,
        target_hidden_states: torch.Tensor,
        token_indices_to_sample: torch.Tensor | None,
        cad: CommonAttentionMetadata,
        num_rejected_tokens_gpu: torch.Tensor | None,
    ) -> tuple[int, torch.Tensor, CommonAttentionMetadata]:
        # The selector scores the first lattice step against the bonus token.
        self._dflash2_anchor_ids = next_token_ids
        self._dflash2_num_reqs = cad.batch_size()
        return super().set_inputs_first_pass(
            target_token_ids=target_token_ids,
            next_token_ids=next_token_ids,
            target_positions=target_positions,
            target_hidden_states=target_hidden_states,
            token_indices_to_sample=token_indices_to_sample,
            cad=cad,
            num_rejected_tokens_gpu=num_rejected_tokens_gpu,
        )

    @override
    def _greedy_sample(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Walk one greedy path through the DFlash2 candidate lattice.

        Replaces the DFlash v1 argmax-over-lm_head draft: hidden_states holds
        [num_reqs * num_speculative_tokens, hidden] sampled positions from the
        single parallel draft forward.
        """
        num_reqs = self._dflash2_num_reqs
        steps = self.num_speculative_tokens
        assert hidden_states.shape[0] == num_reqs * steps, (
            f"DFlash2 selector expects {num_reqs * steps} sampled positions, "
            f"got {hidden_states.shape[0]}."
        )
        hidden = hidden_states.view(num_reqs, steps, -1)

        candidate_ids, unary_logits = self.model.compute_candidates(
            hidden.flatten(0, 1)
        )
        candidate_ids = candidate_ids.view(num_reqs, steps, self.selector_top_k)
        unary_logits = unary_logits.view_as(candidate_ids)
        assert self._dflash2_anchor_ids is not None
        anchor_token_ids = self._dflash2_anchor_ids[:num_reqs]

        scores = self.model.model.candidate_selector(
            candidate_ids,
            unary_logits,
            hidden,
            anchor_token_ids,
        )
        # scores: [num_reqs, steps, top_k (predecessor), top_k (successor)].
        # Row 0 is identical across predecessors (anchor-based), so the walk
        # starts at index 0 exactly like the upstream walk kernel.
        batch_idx = torch.arange(num_reqs, device=hidden_states.device)
        prev = torch.zeros(num_reqs, dtype=torch.int64, device=hidden_states.device)
        draft_tokens = torch.empty(
            num_reqs, steps, dtype=torch.int64, device=hidden_states.device
        )
        for step in range(steps):
            step_scores = scores[batch_idx, step, prev]
            # argmax returns the first maximal index, matching the upstream
            # min(where(score == best)) tie-break.
            idx = step_scores.argmax(dim=-1)
            draft_tokens[:, step] = candidate_ids[batch_idx, step, idx]
            prev = idx
        return draft_tokens.view(-1)
