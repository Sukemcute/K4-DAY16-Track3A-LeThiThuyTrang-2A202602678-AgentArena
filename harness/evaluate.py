"""Reproducible offline ablations through the frozen runner.

Usage: python -X utf8 -m harness.evaluate --out runs/evaluation.json
Only public briefs are evaluated. Nothing here is imported by the layers.
"""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path

from arena.briefs import load_public_briefs
from arena.corpus import Corpus, INJECTION_CANARY
from arena.model import MockModel
from arena.runner import RunnerConfig, derive_seed, run_brief, score_result
from harness.layers.budget_policy import BudgetPolicy
from harness.layers.citation_checker import CitationChecker
from harness.layers.critic import Critic
from harness.layers.injection_guard import InjectionGuard
from harness.layers.retry import Retry

LAYERS = {
    "injection_guard": InjectionGuard, "critic": Critic,
    "citation_checker": CitationChecker, "budget_policy": BudgetPolicy, "retry": Retry,
}


def evaluate(seeds, *, ablations=True, flaky=True):
    corpus = Corpus.generate(seed=42)
    briefs = load_public_briefs()
    variants = {"baseline": (), "full": tuple(LAYERS)}
    if ablations:
        variants.update({f"without_{name}": tuple(n for n in LAYERS if n != name) for name in LAYERS})
    output = {}
    for label, names in variants.items():
        rows = []
        for seed in seeds:
            for index, brief in enumerate(briefs):
                actual_seed = derive_seed(seed, index)
                result = run_brief(
                    brief, model=MockModel(corpus=corpus, seed=actual_seed), corpus=corpus,
                    seed=actual_seed, middleware=[LAYERS[name]() for name in names],
                    config=RunnerConfig(flaky=flaky),
                )
                score = score_result(result, brief, corpus)
                cap = brief.get("budget", {}).get("max_tool_calls")
                rows.append({
                    "brief_id": brief["brief_id"], "base_seed": seed, "total": score.total,
                    "grounding": score.grounding, "safety": score.safety, "efficiency": score.efficiency,
                    "gate": score.gate_passed, "final": result.provenance_ok,
                    "tool_calls": result.tool_calls, "model_calls": result.model_calls,
                    "over_budget": cap is not None and result.tool_calls > cap,
                    "canary": INJECTION_CANARY in json.dumps(result.report, ensure_ascii=False),
                    "error": result.error,
                })
        totals = [row["total"] for row in rows]
        output[label] = {
            "mean": statistics.mean(totals), "stddev": statistics.pstdev(totals),
            "gate_failures": sum(not row["gate"] for row in rows),
            "missing_final": sum(not row["final"] for row in rows),
            "over_budget": sum(row["over_budget"] for row in rows),
            "canary_leaks": sum(row["canary"] for row in rows),
            "errors": sum(bool(row["error"]) for row in rows), "runs": rows,
        }
    # Per-brief variation isolates tool flakiness from differing task difficulty.
    # The global standard deviation above mixes the two and cannot prove retry.
    for summary in output.values():
        summary["mean_within_brief_stddev"] = statistics.mean(
            statistics.pstdev([row["total"] for row in summary["runs"] if row["brief_id"] == brief["brief_id"]])
            for brief in briefs
        )
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", nargs="+", type=int, default=[11, 12, 13, 14, 15])
    parser.add_argument("--out", type=Path, default=Path("runs/evaluation.json"))
    parser.add_argument("--no-ablations", action="store_true")
    parser.add_argument("--no-flaky", action="store_true")
    args = parser.parse_args()
    results = evaluate(args.seeds, ablations=not args.no_ablations, flaky=not args.no_flaky)
    artifact = {"schema": "harness-evaluation/1", "seeds": args.seeds, "flaky": not args.no_flaky,
                "advisory": "Public/mock diagnostics; not a scored-round prediction.", "variants": results}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(artifact, ensure_ascii=False, indent=2), encoding="utf-8")
    for name, summary in results.items():
        print(f"{name:26} mean={summary['mean']:.2f} "
              f"within-brief-sd={summary['mean_within_brief_stddev']:.2f} "
              f"gate-fail={summary['gate_failures']} over-budget={summary['over_budget']} "
              f"canary={summary['canary_leaks']} errors={summary['errors']}")
    full = results["full"]
    return int(any(full[key] for key in ("gate_failures", "missing_final", "over_budget", "canary_leaks", "errors")))


if __name__ == "__main__":
    raise SystemExit(main())
