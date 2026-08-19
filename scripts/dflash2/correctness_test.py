#!/usr/bin/env python3
"""Greedy correctness test for DFlash2 speculative decoding on SM75.

Usage:
  correctness_test.py collect --out results-dflash2.json   # on DFlash2 server
  correctness_test.py collect --out results-ar.json        # on AR baseline
  correctness_test.py compare results-dflash2.json results-ar.json
"""
import argparse
import json
import time

import urllib.request

PROMPTS = [
    "The capital of France is",
    "Explain what a transformer neural network is in two sentences:",
    "def fibonacci(n):\n    \"\"\"Return the n-th Fibonacci number.\"\"\"",
    "Translate to French: The weather is nice today.",
    "List the first ten prime numbers:",
    "The theory of general relativity states that",
    "Write a haiku about autumn leaves.",
    "SELECT name, age FROM users WHERE",
    "In quantum mechanics, the Schrödinger equation",
    "Step-by-step, solve: 17 * 23 =",
]

MAX_TOKENS = 200


def complete(url, model, prompt):
    body = json.dumps(
        {
            "model": model,
            "prompt": prompt,
            "max_tokens": MAX_TOKENS,
            "temperature": 0.0,
            "seed": 0,
        }
    ).encode()
    req = urllib.request.Request(
        url, data=body, headers={"Content-Type": "application/json"}
    )
    t0 = time.time()
    with urllib.request.urlopen(req, timeout=600) as resp:
        data = json.loads(resp.read())
    dt = time.time() - t0
    choice = data["choices"][0]
    return {
        "text": choice["text"],
        "finish_reason": choice["finish_reason"],
        "completion_tokens": data["usage"]["completion_tokens"],
        "latency_s": round(dt, 3),
    }


def collect(args):
    with open("/tmp/dflash2-model-name.txt") as f:
        model = f.read().strip()
    url = f"http://127.0.0.1:{args.port}/v1/completions"
    results = {"model": model, "port": args.port, "prompts": []}
    for i, prompt in enumerate(PROMPTS):
        r = complete(url, model, prompt)
        print(f"[{i + 1}/{len(PROMPTS)}] {r['completion_tokens']} tok, "
              f"{r['latency_s']}s, finish={r['finish_reason']}")
        results["prompts"].append({"prompt": prompt, **r})
    with open(args.out, "w") as f:
        json.dump(results, f, indent=2)
    print(f"saved -> {args.out}")


def compare(args):
    a = json.load(open(args.files[0]))
    b = json.load(open(args.files[1]))
    assert len(a["prompts"]) == len(b["prompts"])
    ok = True
    for pa, pb in zip(a["prompts"], b["prompts"]):
        assert pa["prompt"] == pb["prompt"]
        if pa["text"] == pb["text"] and pa["completion_tokens"] == pb["completion_tokens"]:
            print(f"PASS  ({pa['completion_tokens']:>4} tok)  {pa['prompt'][:60]!r}")
        else:
            ok = False
            ta, tb = pa["text"], pb["text"]
            n = 0
            while n < min(len(ta), len(tb)) and ta[n] == tb[n]:
                n += 1
            print(f"FAIL  {pa['prompt'][:60]!r}")
            print(f"      diverges at char {n}:")
            print(f"      A: ...{ta[max(0, n - 20):n + 60]!r}")
            print(f"      B: ...{tb[max(0, n - 20):n + 60]!r}")
    print("\nRESULT:", "LOSSLESS (greedy outputs identical)" if ok else "MISMATCH")
    return 0 if ok else 1


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("collect")
    c.add_argument("--out", required=True)
    c.add_argument("--port", type=int, default=8000)
    m = sub.add_parser("compare")
    m.add_argument("files", nargs=2)
    args = ap.parse_args()
    if args.cmd == "collect":
        collect(args)
    else:
        raise SystemExit(compare(args))


if __name__ == "__main__":
    main()
