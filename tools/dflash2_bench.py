# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""DFlash2 SM75 benchmark harness.

Streams chat completions against a running vLLM endpoint and records TTFT,
decode tok/s, end-to-end tok/s, and inter-token latencies for four workload
families (coding, reasoning, chat, agent/tool-like). Speculative-decoding
acceptance counters are sampled from the Prometheus /metrics endpoint before
and after each measured run.

Usage:
  python tools/dflash2_bench.py --port 8000 --label dflash2-7 \
      --runs 3 --warmups 3 --max-tokens 1024
"""

import argparse
import json
import re
import statistics
import time
import urllib.request

import requests

WORKLOADS = {
    "coding": (
        "Write a Python module implementing a thread-safe LRU cache with TTL "
        "expiry, plus a C++ snippet showing an equivalent std::list-based LRU, "
        "and a JSON schema describing the cache configuration. Include "
        "docstrings and type hints everywhere.",
        1024,
    ),
    "reasoning": (
        "Solve step by step: A train leaves city A at 09:00 heading to city B "
        "480 km away at 120 km/h. A second train leaves B at 09:30 toward A at "
        "160 km/h. When and where do they meet? Then generalize: derive the "
        "meeting time formula for arbitrary delay d, speeds v1, v2 and "
        "distance D, and verify it on three numeric examples, explaining each "
        "algebraic step in detail.",
        1024,
    ),
    "chat": (
        "Explain to a curious hobbyist how speculative decoding speeds up LLM "
        "inference, using an analogy, then discuss when it helps and when it "
        "does not. Keep a friendly conversational tone.",
        512,
    ),
    "agent": (
        'You are an agent. Emit a JSON tool-call plan (as raw JSON, no prose) '
        'with fields "steps", each step having "tool", "arguments", and '
        '"expected_result", for the task: fetch a GitHub pull request, list '
        'its changed files, run the test suite, and post a summary comment. '
        "Repeat the schema for three different repositories and add a final "
        '"validation" section listing invariants to check.',
        512,
    ),
}

SPEC_METRIC_PATTERNS = [
    "vllm:spec_decode_num_draft_tokens_total",
    "vllm:spec_decode_num_accepted_tokens_total",
    "vllm:spec_decode_num_emitted_tokens_total",
]


def read_spec_metrics(base_url: str) -> dict[str, float]:
    try:
        text = requests.get(f"{base_url}/metrics", timeout=10).text
    except requests.RequestException:
        return {}
    out: dict[str, float] = {}
    for line in text.splitlines():
        for pat in SPEC_METRIC_PATTERNS:
            if line.startswith(pat) and not line.startswith(pat + "_"):
                parts = line.split()
                if len(parts) == 2:
                    try:
                        out[pat] = float(parts[1])
                    except ValueError:
                        pass
    return out


def stream_request(
    base_url: str,
    model: str,
    prompt: str,
    max_tokens: int,
    temperature: float,
) -> dict:
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": temperature,
        "top_p": 1.0,
        "stream": True,
        "stream_options": {"include_usage": True},
        "ignore_eos": True,
    }
    t0 = time.perf_counter()
    ttft = None
    itls: list[float] = []
    chunks = 0
    usage = None
    text_parts: list[str] = []
    last = t0
    with requests.post(
        f"{base_url}/chat/completions",
        json=payload,
        stream=True,
        timeout=600,
    ) as resp:
        resp.raise_for_status()
        for raw in resp.iter_lines():
            if not raw or not raw.startswith(b"data: "):
                continue
            data = raw[6:]
            if data.strip() == b"[DONE]":
                break
            obj = json.loads(data)
            if obj.get("usage"):
                usage = obj["usage"]
            delta = (obj.get("choices") or [{}])[0].get("delta", {})
            piece = delta.get("content")
            if piece:
                now = time.perf_counter()
                if ttft is None:
                    ttft = now - t0
                else:
                    itls.append(now - last)
                last = now
                chunks += 1
                text_parts.append(piece)
    total = time.perf_counter() - t0
    completion_tokens = usage["completion_tokens"] if usage else chunks
    decode_time = total - (ttft or 0.0)
    return {
        "ttft": ttft or 0.0,
        "e2e": total,
        "tokens": completion_tokens,
        "decode_tok_s": (completion_tokens - 1) / decode_time if decode_time > 0 else 0.0,
        "e2e_tok_s": completion_tokens / total if total > 0 else 0.0,
        "itl_mean_ms": statistics.mean(itls) * 1000 if itls else 0.0,
        "itl_p50_ms": statistics.median(itls) * 1000 if itls else 0.0,
        "text": "".join(text_parts),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--model", default=None, help="served model name")
    ap.add_argument("--label", default="run")
    ap.add_argument("--runs", type=int, default=3)
    ap.add_argument("--warmups", type=int, default=3)
    ap.add_argument("--max-tokens", type=int, default=0, help="override workload default")
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--workloads", nargs="+", default=list(WORKLOADS))
    ap.add_argument("--save-texts", default=None, help="dump generated texts here")
    args = ap.parse_args()

    base = f"http://{args.host}:{args.port}/v1"
    if args.model is None:
        models = requests.get(f"{base}/models", timeout=10).json()
        args.model = models["data"][0]["id"]
    print(f"label={args.label} model={args.model} temperature={args.temperature}")

    results: dict[str, list[dict]] = {}
    texts: dict[str, str] = {}
    for name in args.workloads:
        prompt, default_tokens = WORKLOADS[name]
        max_tokens = args.max_tokens or default_tokens
        print(f"\n=== workload={name} max_tokens={max_tokens} ===")
        for w in range(args.warmups):
            r = stream_request(base, args.model, prompt, max_tokens, args.temperature)
            print(f"  warmup {w}: {r['tokens']} tok, {r['decode_tok_s']:.1f} tok/s decode")
        for i in range(args.runs):
            before = read_spec_metrics(f"http://{args.host}:{args.port}")
            r = stream_request(base, args.model, prompt, max_tokens, args.temperature)
            after = read_spec_metrics(f"http://{args.host}:{args.port}")
            spec = {}
            if before and after:
                draft = after.get(SPEC_METRIC_PATTERNS[0], 0) - before.get(
                    SPEC_METRIC_PATTERNS[0], 0
                )
                accepted = after.get(SPEC_METRIC_PATTERNS[1], 0) - before.get(
                    SPEC_METRIC_PATTERNS[1], 0
                )
                spec = {
                    "draft_tokens": draft,
                    "accepted_tokens": accepted,
                    "acceptance_rate": accepted / draft if draft else 0.0,
                    "mean_accepted_len": (accepted + 1) / max(1, draft / 7 if draft else 1),
                }
            r["spec"] = spec
            results.setdefault(name, []).append(r)
            texts[name] = r["text"]
            print(
                f"  run {i}: ttft={r['ttft']*1000:.0f}ms decode={r['decode_tok_s']:.1f} "
                f"tok/s e2e={r['e2e_tok_s']:.1f} tok/s itl_mean={r['itl_mean_ms']:.1f}ms "
                f"tokens={r['tokens']} spec={spec}"
            )

    summary = {"label": args.label, "model": args.model}
    for name, runs in results.items():
        summary[name] = {
            "decode_tok_s": statistics.mean(r["decode_tok_s"] for r in runs),
            "e2e_tok_s": statistics.mean(r["e2e_tok_s"] for r in runs),
            "ttft_ms": statistics.mean(r["ttft"] for r in runs) * 1000,
            "itl_mean_ms": statistics.mean(r["itl_mean_ms"] for r in runs),
            "acceptance_rate": statistics.mean(
                r["spec"].get("acceptance_rate", 0.0) for r in runs
            ),
        }
    print("\n=== SUMMARY ===")
    print(json.dumps(summary, indent=2))
    if args.save_texts:
        with open(args.save_texts, "w") as f:
            json.dump({"label": args.label, "texts": texts}, f, ensure_ascii=False)
        print(f"texts saved to {args.save_texts}")


if __name__ == "__main__":
    main()
