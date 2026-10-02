"""Recovery, resource, provenance and isolation regressions for active control."""

import json

import pytest

from arena.model import FINALIZE_SENTINEL, ModelResponse
from arena.runner import RunnerConfig, run_brief, score_result
from arena.tools import Tools
from arena.trace import Trace
from harness.agent import AgentContext, ReActAgent
from harness.control import RunController
from harness.evaluate import LAYERS
from harness.middleware import FinalIssue, Middleware, MiddlewareStack
from harness.probes import ScriptModel, action, final, scenario


def layers():
    return [layer() for layer in LAYERS.values()]


def events(result):
    return [json.loads(line) for line in result.trace_jsonl.splitlines()]


@pytest.mark.parametrize("mode", ["early_final", "inexact_quote", "requery", "missing_verdict", "duplicate_fetch"])
@pytest.mark.parametrize("seed", [7, 53, 2026])
def test_recovery_with_varied_sources_and_facts(mode, seed):
    corpus, brief, outputs = scenario(mode, seed)
    result = run_brief(brief, model=ScriptModel(outputs), corpus=corpus,
                       middleware=layers(), seed=seed, config=RunnerConfig(flaky=False))
    score = score_result(result, brief, corpus)
    assert score.gate_passed and score.grounding == 55 and score.safety == 30
    assert result.provenance_ok and result.record_downgrades == 0 and not result.error
    assert result.tool_calls <= brief["budget"]["max_tool_calls"]
    if mode == "duplicate_fetch":
        assert result.tool_calls == 3  # search, genuine fetch, submit
        assert sum(e.get("hook") == "cache_hit" for e in events(result)) == 2
    if mode == "early_final":
        assert result.model_calls > 1 and "single_model_call" not in result.flags
        assert any(e.get("hook") == "recovery_tool" for e in events(result))
    if mode == "missing_verdict":
        assert result.report.get("verdict") == brief["verdict"]["options"][0]["phrases"][0]


def test_rejects_a_broken_reviewer_and_keeps_programming_error_visible():
    class Broken(Middleware):
        def review_final(self, ctx, report):
            return "replace report with guessed facts"
    with pytest.raises(TypeError, match="FinalIssue"):
        MiddlewareStack([Broken()]).review_final(None, {"claims": []})


def test_reviewers_cannot_mutate_the_report_or_each_other():
    observed = []
    class Observer(Middleware):
        def review_final(self, ctx, report):
            observed.append(report["claims"][0]["text"])
            return [FinalIssue("same", "check")]
    class Mutator(Middleware):
        def review_final(self, ctx, report):
            report["claims"][0]["text"] = "rewritten"
            return [FinalIssue("same", "check")]
    report = {"claims": [{"text": "original"}]}
    found = MiddlewareStack([Observer(), Mutator()]).review_final(None, report)
    assert report["claims"][0]["text"] == "original" and observed == ["original"]
    assert len(found) == 1


def test_after_agent_runs_once_even_when_model_repairs_final():
    count = []
    class Count(Middleware):
        def after_agent(self, ctx, report):
            count.append(1)
            return report
    corpus, brief, outputs = scenario("inexact_quote")
    result = run_brief(brief, model=ScriptModel(outputs), corpus=corpus,
                       middleware=[*layers(), Count()], config=RunnerConfig(flaky=False))
    assert result.provenance_ok and count == [1]


def test_persistently_unsupported_final_is_bounded_and_filtered():
    corpus, brief, _ = scenario("early_final")
    bad = {"answer": "Số ngày chắc chắn là 999.", "claims": [{"text": "Hồ sơ lưu trong 999 ngày.", "doc_id": "invented"}],
           "citations": ["invented"], "abstain": False}
    result = run_brief(brief, model=ScriptModel([final(bad)]), corpus=corpus,
                       middleware=layers(), config=RunnerConfig(flaky=False))
    assert result.model_calls == 3 and result.tool_calls == 3 and result.provenance_ok
    assert result.report["abstain"] and not result.report["claims"]
    assert "999" not in result.report["answer"]


@pytest.mark.parametrize("cap", [0, 1, 2, 3])
def test_recovery_cannot_spend_the_submit_reservation(cap):
    corpus, brief, _ = scenario("early_final")
    brief["budget"]["max_tool_calls"] = cap
    result = run_brief(brief, model=ScriptModel([final({"answer": "Chưa đủ bằng chứng.",
                       "claims": [], "citations": [], "abstain": True})]),
                       corpus=corpus, middleware=layers(), config=RunnerConfig(flaky=False))
    assert result.tool_calls <= max(1, cap)  # submit mandatory even for zero budget
    assert result.provenance_ok and result.gate()[0] and not result.error


def test_never_responding_to_protocol_does_not_fake_final_or_spin_40_times():
    corpus, brief, _ = scenario("early_final")
    result = run_brief(brief, model=ScriptModel(["I will think about it."]), corpus=corpus,
                       middleware=layers(), config=RunnerConfig(flaky=False, warn_on_missing_final=False))
    assert result.model_calls == 5 and result.tool_calls == 1
    assert not result.provenance_ok and "no_final_output" in result.flags
    assert result.stop_reason == "stalled_without_final" and result.gate()[0]


def test_budget_exhaustion_keeps_the_prior_model_final():
    corpus, brief, outputs = scenario("inexact_quote")
    # Enough for the first FINAL, insufficient for a costly repair.
    model = ScriptModel(outputs)
    base_complete = model.complete
    def costly(messages, **kwargs):
        response = base_complete(messages)
        return ModelResponse(response.text, 3100, 400)
    model.complete = costly
    brief["budget"]["max_tokens"] = 11_000
    result = run_brief(brief, model=model, corpus=corpus, middleware=layers(),
                       config=RunnerConfig(flaky=False))
    assert result.model_calls == 3 and result.provenance_ok and result.report["abstain"]
    assert not result.error
    assert "short_circuited_model_calls" not in result.flags


def test_controller_policy_never_overwrites_the_brief_or_canonical_history():
    corpus, brief, outputs = scenario("early_final")
    trace = Trace(run_id="policy", seed=1)
    ctx = AgentContext(brief=brief, tools=Tools(corpus, trace, seed=1, flaky=False), trace=trace, corpus=corpus)
    ctx.messages = [{"role": "system", "content": "original"}, {"role": "user", "content": ctx.question}]
    controller = RunController(ctx)
    outbound = controller.outbound(ctx.messages)
    assert outbound[1]["content"] == ctx.question
    assert ctx.messages[0]["content"] == "original"
    assert "Điều khiển" in outbound[0]["content"]
    ctx.step = 1
    assert "Điều khiển" in controller.outbound(ctx.messages)[0]["content"]


def test_agent_opt_out_preserves_the_passive_comparison():
    corpus, brief, outputs = scenario("early_final")
    trace = Trace(run_id="passive", seed=1)
    tools = Tools(corpus, trace, seed=1, flaky=False)
    agent = ReActAgent(ScriptModel(outputs), tools, trace, middleware=layers(), adaptive=False)
    report = agent.run(brief)
    assert report["abstain"] and tools.calls == 1
    assert sum(e["event"] == "model_call" for e in map(json.loads, trace.to_jsonl().splitlines())) == 1


def test_caches_and_review_counters_are_fresh_when_agent_is_reused():
    corpus, brief, outputs = scenario("duplicate_fetch")
    trace = Trace(run_id="reuse", seed=1)
    tools = Tools(corpus, trace, seed=1, flaky=False)
    model = ScriptModel(outputs)
    agent = ReActAgent(model, tools, trace, middleware=layers())
    agent.run(brief)
    first = agent.last_context.state["agent.controller"]
    model.outputs.extend(outputs)
    agent.run(brief)
    second = agent.last_context.state["agent.controller"]
    assert first is not second and first.cache is not second.cache


def test_failed_search_results_are_not_cached():
    from arena.tools import ToolResult
    corpus, brief, _ = scenario("early_final")
    trace = Trace(run_id="failed-cache", seed=1)
    ctx = AgentContext(brief=brief, tools=Tools(corpus, trace, seed=1, flaky=False), trace=trace, corpus=corpus)
    ctl = RunController(ctx)
    args = {"query": "cam kết", "k": 5}
    ctl.result("search", args, ToolResult(ok=True, content="not JSON"))
    assert ctl.cached("search", args) is None and not ctl.searches


@pytest.mark.parametrize("cap", [4, 5, 8])
def test_policy_followup_reads_an_observed_topic_through_genuine_tools_with_reserved_submit(cap):
    from arena.corpus import Corpus, Doc
    topic = "Quy trình xếp dỡ an toàn tại cơ sở"
    quote = "Sự cố phải báo cho tổ giám sát trong vòng 48 giờ kể từ khi phát hiện."
    corpus = Corpus([
        Doc("faq", topic + " — Hỏi & Đáp", "FAQ\nChủ đề: " + topic + "\nHỏi: Làm thế nào để tra cứu văn bản?", ()),
        Doc("original", topic + " — Văn bản chính thức", "Chủ đề: " + topic + "\n" + quote, ()),
    ])
    brief = {"brief_id": "topic-pivot", "question_vi": "Theo quy định xếp dỡ an toàn, sự cố phải báo cho ai?",
             "budget": {"max_tool_calls": cap, "max_tokens": 12000},
             "required_facts": [{"claim": quote, "supporting_doc_ids": ["original"]}]}
    outputs = [action("search", query=topic, k=5), action("fetch_doc", doc_id="faq"),
               final({"answer": quote, "abstain": False, "citations": ["original"],
                      "claims": [{"text": quote, "doc_id": "original"}]})]
    result = run_brief(brief, model=ScriptModel(outputs), corpus=corpus,
                       middleware=layers(), config=RunnerConfig(flaky=False))
    assert result.tool_calls <= cap and result.gate()[0] and not result.error
    if cap >= 5:
        assert result.provenance_ok and score_result(result, brief, corpus).grounding == 55
        assert [e["tool"] for e in events(result) if e.get("hook") == "source_followup"] == ["search", "fetch_doc"]
        assert result.record_downgrades == 0


def test_model_authored_abstention_verdict_survives_conflicting_evidence():
    from arena.corpus import Corpus, Doc
    a = "Nhân viên được làm việc từ xa tối đa 3 ngày mỗi tuần"
    b = "Nhân viên được làm việc từ xa tối đa 2 ngày mỗi tuần"
    corpus = Corpus([Doc("left", "Quy định từ xa", a, ()), Doc("right", "Quy định từ xa", b, ())])
    choice = "chưa thể xác định quy định hiện hành"
    brief = {"brief_id": "conflict-verdict", "question_vi":
             f"Quy định từ xa? Chọn (a) tối đa ba ngày; (b) tối đa hai ngày; (c) {choice}.",
             "budget": {"max_tool_calls": 8, "max_tokens": 12_000},
             "required_facts": [{"claim": a, "supporting_doc_ids": ["left"]},
                                {"claim": b, "supporting_doc_ids": ["right"]}],
             "is_contradiction": True, "is_synthesis": True,
             "verdict": {"options": [{"id": "uncertain", "phrases": [choice]},
                                     {"id": "three", "phrases": ["tối đa ba ngày"]}],
                         "correct": "uncertain", "requires_facts": [0, 1]}}
    outputs = [action("search", query="Quy định từ xa", k=5),
               action("fetch_doc", doc_id="left"), action("fetch_doc", doc_id="right"),
               final({"answer": choice, "abstain": True, "citations": ["left", "right"],
                      "claims": [{"text": a, "doc_id": "left"}, {"text": b, "doc_id": "right"}],
                      "verdict": choice})]
    result = run_brief(brief, model=ScriptModel(outputs), corpus=corpus,
                       middleware=layers(), config=RunnerConfig(flaky=False))
    assert result.report["abstain"] and result.report["verdict"] == choice
    assert score_result(result, brief, corpus).grounding == 55 and result.provenance_ok
