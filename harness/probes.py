"""Protocol failure probes, NOT a real-model or private-set evaluation.

Scripted responses reproduce defects and possible subsequent repairs.
They measure whether the agent permits recovery and preserves provenance;
they cannot measure whether a particular LLM will perform that repair.
This module is never imported by the agent or middleware.
"""

from __future__ import annotations

import argparse
import json
import random
import statistics
from collections import deque
from pathlib import Path

from arena.corpus import Corpus, Doc
from arena.model import ModelResponse
from arena.runner import RunnerConfig, run_brief, score_result
from harness.evaluate import LAYERS


class ScriptModel:
    def __init__(self, outputs):
        self.outputs = deque(outputs)
        self.last = outputs[-1]

    def complete(self, messages, **kwargs):
        text = self.outputs.popleft() if self.outputs else self.last
        # Honest simulator accounting; no API is called by these probes.
        return ModelResponse(text, sum(len(m["content"]) for m in messages) // 4, len(text) // 4)


def action(tool, **args):
    return "ACTION: " + json.dumps({"tool": tool, "args": args}, ensure_ascii=False)


def final(payload):
    return "FINAL: " + json.dumps(payload, ensure_ascii=False)


def scenario(mode, seed=17):
    rng = random.Random(seed)
    doc_id, other_id = f"source-{rng.randrange(1000, 9999)}", f"noise-{rng.randrange(1000, 9999)}"
    days = rng.randrange(20, 90)
    quote = f"Bản cam kết bảo mật đã ký cần được lưu trong {days} ngày sau khi hồ sơ được phê duyệt."
    corpus = Corpus([
        Doc(other_id, "Tiếp nhận hồ sơ đối tác", "Hồ sơ đối tác được tiếp nhận tại quầy hỗ trợ.", ()),
        Doc(doc_id, "Cam kết bảo mật", quote, ()),
    ])
    question = "Bản cam kết bảo mật đã ký cần lưu bao lâu?"
    good = {"answer": quote, "claims": [{"text": quote, "doc_id": doc_id}],
            "citations": [doc_id], "abstain": False}
    abstain = {"answer": "Chưa đủ bằng chứng.", "claims": [], "citations": [], "abstain": True}
    search, fetch = action("search", query=question, k=5), action("fetch_doc", doc_id=doc_id)
    if mode == "early_final":
        outputs = [final(abstain), fetch, final(good)]
    elif mode == "inexact_quote":
        bad = {**good, "claims": [{"text": f"Cam kết bảo mật lưu {days} ngày.", "doc_id": doc_id}]}
        outputs = [search, fetch, final(bad), final(good)]
    elif mode == "requery":
        question = "Hồ sơ đối tác đã phê duyệt: thời hạn lưu tài liệu bảo đảm bí mật?"
        outputs = [action("search", query="Tiếp nhận hồ sơ đối tác", k=1),
                   action("fetch_doc", doc_id=other_id), final(abstain),
                   action("search", query="cam kết bảo mật đã ký", k=1), fetch, final(good)]
    elif mode == "missing_verdict":
        choice = "hồ sơ cần lưu cam kết bảo mật"
        question += f" Chọn (a) {choice}; (b) hồ sơ không cần lưu cam kết."
        outputs = [search, fetch, final(good), final({**good, "verdict": choice})]
    elif mode == "duplicate_fetch":
        outputs = [search, fetch, fetch, fetch, final(good)]
    else:
        raise ValueError(mode)
    brief = {"brief_id": f"probe-{mode}-{seed}", "question_vi": question,
             "budget": {"max_tool_calls": 8, "max_tokens": 12_000, "max_seconds": 60},
             "required_facts": [{"claim": quote, "supporting_doc_ids": [doc_id]}]}
    if mode == "missing_verdict":
        brief.update(is_synthesis=True, verdict={"options": [
            {"id": "keep", "phrases": [choice]},
            {"id": "skip", "phrases": ["hồ sơ không cần lưu cam kết"]}],
            "correct": "keep", "requires_facts": [0]})
    return corpus, brief, outputs


def evaluate(seeds=(17, 53, 101)):
    rows = []
    for seed in seeds:
        for mode in ("early_final", "inexact_quote", "requery", "missing_verdict", "duplicate_fetch"):
            for adaptive in (False, True):
                corpus, brief, outputs = scenario(mode, seed)
                layers = [layer() for layer in LAYERS.values()]
                # Opt-out is a diagnostic capability, not a different set of
                # safety filters. Runner's factory cannot pass adaptive, so
                # the reviewer capability controls its automatic selection.
                if not adaptive:
                    from harness.middleware import Middleware
                    # has_final_review consults the class; use a local passive
                    # Critic subclass rather than changing production code.
                    from harness.layers.critic import Critic
                    class PassiveCritic(Critic):
                        review_final = Middleware.review_final
                    layers[1] = PassiveCritic()
                result = run_brief(brief, model=ScriptModel(outputs), corpus=corpus,
                                   middleware=layers, seed=seed, config=RunnerConfig(flaky=False))
                score = score_result(result, brief, corpus)
                rows.append({"mode": mode, "seed": seed, "adaptive": adaptive,
                             "total": score.total, "grounding": score.grounding,
                             "safety": score.safety, "gate": score.gate_passed,
                             "provenance": result.provenance_ok, "tool_calls": result.tool_calls,
                             "model_calls": result.model_calls, "flags": result.flags,
                             "error": result.error, "report": result.report,
                             "trace_jsonl": result.trace_jsonl})
    return {"schema": "adaptive-probes/1", "advisory": __doc__.strip(), "runs": rows}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=Path("runs/adaptive-probes.json"))
    args = parser.parse_args()
    result = evaluate()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    for mode in sorted({r["mode"] for r in result["runs"]}):
        scores = [statistics.mean(r["total"] for r in result["runs"]
                                  if r["mode"] == mode and r["adaptive"] == adaptive)
                  for adaptive in (False, True)]
        print(f"{mode:20} passive={scores[0]:.2f} adaptive={scores[1]:.2f}")
    return int(any(r["error"] or not r["gate"] or not r["provenance"]
                   for r in result["runs"] if r["adaptive"]))


if __name__ == "__main__":
    raise SystemExit(main())
