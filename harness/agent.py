"""ReAct agent with bounded, evidence-driven FINAL review.

STUDENT-OWNED. The baseline loop remains available with adaptive=False
or without a reviewing middleware. With Critic installed, the controller:

* asks for retrieval before accepting an unsupported early FINAL;
* bootstraps real search/fetch when the model skipped retrieval;
* lets the model repair bad quotations, missing evidence or verdict;
* reuses successful, sanitised read results and detects stalled loops;
* retains a model-authored FINAL when resources prevent another repair.

All model calls still go through _call_model and the frozen runner. All
real tool calls go through the normal middleware. No model/tool trace is
fabricated. Parsing uses arena.model.parse_output with the scorer's
canonicaliser; only raw model FINAL text supplies claim provenance.
MAX_STEPS remains 40, after_agent runs once, and submit bypasses retry.

The compact control policy applies automatically with a reviewer, also
when the runner uses its default prompt. REAL_MODEL_PROMPT_ADDENDUM is
an optional, more detailed protocol guide for real endpoints. Measure its
additional token cost; it is not required to activate adaptive control.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from arena.model import (
    ARENA_SYSTEM_PROMPT,
    TOOL_ERROR_PREFIX,
    parse_output,
)
from arena.tools import ToolResult

from harness.middleware import Middleware, MiddlewareStack
from harness.control import RunController

#: Hard ceiling on model turns. >= 40 is a REQUIREMENT, not a taste: with
#: every tool call returning noise the mock needs 31 turns to reach its
#: FINAL, and a run that hits the cap produces no report and scores zero
#: with no error message anywhere.
MAX_STEPS = 40

#: `k` a search is allowed to ask for. The mock asks for 5; the clamp is
#: here so a bug (or a creative prompt) cannot pull the whole corpus into
#: one observation and drown the context.
MAX_SEARCH_K = 20

#: Keys that make a decoded payload a REPORT rather than something the
#: model merely quoted. Normalisation is deliberately generous about what
#: counts as a FINAL marker (`final:`, `**Final:**`, `### FINAL`, indented,
#: quoted), so a stray line of prose whose tail happens to decode as JSON
#: can manufacture an empty "report" and end the run on turn one. A
#: payload carrying none of these keys is not a report.
#:
#: THE KEYS ARE NOT ENOUGH ON THEIR OWN, and this is measured, not
#: theoretical: `ARENA_SYSTEM_PROMPT`'s own template line carries ALL
#: FOUR, so a model that restates the required format — an ordinary thing
#: for a model to do on turn one — walks straight through a keys-only
#: check and ends the run with the TEMPLATE as its report while a perfect
#: ACTION sits underneath. Swept over four payload shapes x three
#: positions x three turns on the trap-spanning set: 1080 of 1080 runs
#: ended on the quoted turn, the ellipsis form wiping every one of them
#: (92.52 -> 0.00). With the content check below: 0 of 1080 for every
#: placeholder shape. Hence `_is_report_payload`, which also asks whether
#: the payload carries CONTENT.
REPORT_KEYS = ("answer", "claims", "abstain", "citations")

#: How many times ONE RUN may put a FINAL aside because the model wrote a
#: well-formed ACTION underneath it. Bounded on purpose: a model that
#: appends an ACTION to every FINAL would otherwise never be allowed to
#: finish. After this many deferrals the FINAL is taken at face value.
MAX_FINAL_DEFERRALS = 2

#: What a model writes where CONTENT belongs when it is QUOTING the
#: protocol instead of answering: the template's own `...`, an ellipsis,
#: a dash, or an `<angle-bracket slot>`.
_PLACEHOLDER_RE = re.compile(r"\A[\s.…·\-–—]*\Z")

#: The frozen parser's FINAL marker is `^FINAL:` at a line start
#: (`arena.model._FINAL_RE`). Used ONLY to locate marker lines — every
#: payload on this path is still decoded by `parse_output` itself.
_FINAL_MARKER = "FINAL:"

# ---------------------------------------------------------------------------
# The real-model prompt addendum
# ---------------------------------------------------------------------------

#: Appended to `ARENA_SYSTEM_PROMPT` for the scored, real-model path.
#:
#: THREE CLAUSES, EACH ANSWERING A MEASURED FAILURE:
#:
#: A. **Search before abstaining.** gpt-5.6-luna abstained on turn 1 with
#:    zero tool calls on 4 of 6 live runs; the frozen prompt tells the
#:    model to abstain when evidence is absent and never says it has to
#:    look first. On a DEPTH-conforming brief the answer is deliberately
#:    NOT in the question's own top-5, so "searched once, missed, gave
#:    up" is the single likeliest way an honest run lands on the floor.
#:    The clause therefore also demands the RE-QUERY, which is the skill
#:    the private set grades.
#:
#: B. **Strict JSON on the marker's own line.** The frozen `parse_output`
#:    wants `^FINAL:` followed by one decodable object; pretty-printed
#:    payloads, fenced blocks, `**FINAL:**` and smart quotes are ordinary
#:    real-model output and each one costs all 55 grounding points
#:    silently. `arena.scorer._canonicalise_output` repairs many of them,
#:    but not emitting them is cheaper than repairing them.
#:
#: C. **The schema in WORDS, with no quotable template.** This is the
#:    measured one and it is why the text below contains no JSON literal
#:    and no line beginning with the FINAL marker. `ARENA_SYSTEM_PROMPT`
#:    shows the model a filled-in example that is itself valid JSON
#:    carrying all four report keys, and a model that restates the
#:    required format — an ordinary thing to do on turn one — produces a
#:    SHADOW FINAL that ends the run with the template as its report:
#:    grounding 0.00, total 40.15 through the real agent. `_parse` below
#:    defends against it; a prompt with nothing to quote removes the
#:    ammunition instead.
#:
#: Written in Vietnamese because the whole protocol is, and because a
#: Vietnamese instruction is what keeps a Vietnamese answer on-language.
REAL_MODEL_PROMPT_ADDENDUM = """PHỤ LỤC GIAO THỨC — BẮT BUỘC. Nếu có mâu thuẫn, phụ lục này thắng.

A. PHẢI TÌM TRƯỚC KHI ĐƯỢC PHÉP NÓI "KHÔNG ĐỦ CĂN CỨ".
   Lượt đầu tiên của bạn luôn luôn là một ACTION gọi search. Không được kết
   luận ở lượt đầu tiên trong bất kỳ trường hợp nào.
   Chỉ được đặt abstain thành đúng (true) sau khi đã gọi search ít nhất một
   lần VÀ đã gọi fetch_doc ít nhất một lần để đọc toàn văn.
   Câu hỏi thường KHÔNG dùng cùng từ ngữ với tài liệu chứa câu trả lời. Nếu
   kết quả tìm kiếm đầu tiên không chứa câu trả lời, bạn PHẢI diễn đạt lại
   truy vấn bằng thuật ngữ nội bộ (tên quy trình, tên chính sách, tên loại
   văn bản, tên phòng ban) và tìm lại ít nhất một lần nữa trước khi kết luận
   là không có bằng chứng.
   Kết luận "không đủ căn cứ" khi chưa đọc toàn văn tài liệu nào là câu trả
   lời SAI, kể cả khi bạn tin là mình không biết.

B. DÒNG KẾT LUẬN.
   Dòng kết luận phải bắt đầu ngay từ ký tự đầu tiên của dòng bằng nhãn viết
   hoa FINAL: (năm chữ cái in hoa và một dấu hai chấm), rồi đến MỘT đối tượng
   JSON duy nhất nằm TRÊN CÙNG MỘT DÒNG với nhãn đó.
   Không xuống dòng bên trong JSON. Không thụt đầu dòng. Không bọc trong dấu
   nháy ngược hay khối mã. Không in đậm nhãn. Chỉ dùng dấu nháy kép thẳng
   ASCII, không dùng nháy cong. Không có dấu phẩy thừa. Sau dòng kết luận
   không viết thêm bất cứ ký tự nào.

C. NỘI DUNG ĐỐI TƯỢNG JSON — MÔ TẢ BẰNG LỜI, KHÔNG CÓ MẪU ĐỂ CHÉP.
   Đối tượng có bốn khóa bắt buộc, tên khóa viết thường (có thể thêm một khóa
   thứ năm tuỳ chọn — xem mục F):
     - một khóa tên answer, giá trị là chuỗi tiếng Việt trả lời thẳng câu hỏi,
       dưới 600 ký tự;
     - một khóa tên citations, giá trị là mảng các chuỗi mã tài liệu;
     - một khóa tên abstain, giá trị luận lý đúng hoặc sai (không phải chuỗi);
     - một khóa tên claims, giá trị là mảng tối đa bốn phần tử, mỗi phần tử là
       một đối tượng có đúng hai khóa: một khóa tên text chứa câu trích và một
       khóa tên doc_id chứa mã của chính tài liệu chứa câu trích đó.
   Mã tài liệu luôn có dạng doc- rồi ĐÚNG BỐN CHỮ SỐ, ví dụ doc-0004. Không
   tự bịa mã, không rút gọn thành doc-4.
   Tuyệt đối không chép lại phần mô tả định dạng này vào câu trả lời.

D. MỖI PHẦN TỬ claims LÀ MỘT CÂU CHÉP NGUYÊN VĂN.
   Chép đúng từng ký tự một đoạn nằm gọn TRONG MỘT DÒNG của tài liệu bạn đã
   đọc bằng fetch_doc. Không thêm dấu chấm ở cuối, không đổi dấu nháy, không
   sửa chính tả, không ghép hai dòng lại, không tóm tắt, không diễn giải.
   Nếu cần ngắn hơn, chỉ được CẮT BỚT ở hai đầu; phần giữ lại vẫn phải nguyên
   văn. Mỗi câu trích không quá 400 ký tự. Cắt bớt là hợp lệ, viết lại thì mất
   điểm.

E. KẾT THÚC SỚM.
   Mỗi lượt chỉ gọi đúng một công cụ. Không lặp lại một truy vấn đã dùng, không
   gọi lại fetch_doc cho tài liệu đã đọc. Ngay khi đã đọc được tài liệu chứa
   câu trả lời, hãy viết dòng kết luận ở lượt kế tiếp.

F. KHI CÂU HỎI YÊU CẦU CHỌN MỘT KẾT LUẬN.
   Nếu câu hỏi liệt kê sẵn vài phương án đánh chữ cái trong ngoặc — (a), (b), (c) —
   và yêu cầu chọn một, đối tượng JSON có thêm khóa thứ năm tên verdict: giá trị là
   MỘT chuỗi duy nhất, chép nguyên văn đúng từng chữ phương án đã chọn từ câu hỏi,
   không diễn giải lại. Chỉ chọn ĐÚNG MỘT; đưa nhiều hơn một phương án vào verdict
   bị coi là chưa quyết định gì cả. Trường answer vẫn phải trả lời đầy đủ câu hỏi
   như bình thường. Câu hỏi không liệt kê phương án nào thì bỏ hẳn khóa verdict."""


def real_model_system_prompt(base: str = ARENA_SYSTEM_PROMPT) -> str:
    """`base` with `REAL_MODEL_PROMPT_ADDENDUM` appended.

    A function rather than a constant so a student (or the frozen runner)
    can extend a prompt of their own the same way.
    """
    return base.rstrip() + "\n\n" + REAL_MODEL_PROMPT_ADDENDUM.strip() + "\n"


#: `ARENA_SYSTEM_PROMPT` + the addendum. What the SCORED, REAL-MODEL path
#: must pass as `system_prompt`; not the default (see the module
#: docstring for the measured reason).
ARENA_SYSTEM_PROMPT_REAL = real_model_system_prompt()

#: `output_text` is clamped to this before it is stamped on `model_call`.
#: `Trace.emit` truncates any record over 90,000 characters, and a
#: truncated FINAL stops being decodable JSON — which costs all 55
#: grounding points with the gate still passing, i.e. silently. Ordinary
#: output is three orders of magnitude below this.
MAX_OUTPUT_TEXT_CHARS = 60_000


def _canonicalise(text: str) -> str:
    """Rewrite a real endpoint's FINAL into the shape `parse_output` wants.

    Delegates to `arena.scorer._canonicalise_output`, which exists for
    exactly this purpose ("Kept as the single-payload view of
    `_final_payloads`, for Task 6/9, which must recover the report the
    same way the scorer credits it"). It only RESHAPES — indentation,
    fenced code blocks, `**FINAL:**`, a BOM, curly quotes, a trailing
    comma, a payload on the next line — and then the frozen
    `parse_output` does the actual parsing. That is the difference
    between normalising and writing your own parser, and it is the
    difference between 92 and 40.

    Falls back to the raw text if the scorer is not importable, so the
    harness never depends on the grader being present at runtime.
    """
    try:
        from arena.scorer import _canonicalise_output
    except Exception:  # pragma: no cover - the scorer ships with the lab
        return text
    try:
        return _canonicalise_output(text)
    except Exception:  # pragma: no cover - defensive only
        return text


def _is_placeholder(value) -> bool:
    """Is this string a slot the model never filled in?

    `"..."`, `"…"`, `"—"`, `"<câu trả lời>"`, `""` and a missing value all
    say the same thing: the model wrote the SHAPE of an answer, not an
    answer.
    """
    if not isinstance(value, str):
        return True
    stripped = value.strip()
    if not stripped:
        return True
    if _PLACEHOLDER_RE.match(stripped) is not None:
        return True
    return stripped.startswith("<") and stripped.endswith(">")


def _is_report_payload(payload) -> bool:
    """Is this decoded FINAL payload a REPORT, or a quoted example?

    Two questions, and both have to be answered yes:

    1. Does it carry at least one of `REPORT_KEYS`? (A stray line of
       prose whose tail decodes as JSON does not.)
    2. Does it carry CONTENT — one claim with real text, or a real
       `answer`? (The protocol template does not: every content slot in
       it is the literal `"..."`.)

    A payload that fails (2) is worth nothing to the scorer even if it is
    submitted — an empty or placeholder answer scores 0.00 — so refusing
    it can only buy the model another turn, never cost a real report.
    """
    if not isinstance(payload, dict):
        return False
    if not any(key in payload for key in REPORT_KEYS):
        return False
    claims = payload.get("claims")
    if isinstance(claims, list):
        for claim in claims:
            if isinstance(claim, dict) and not _is_placeholder(claim.get("text")):
                return True
    return not _is_placeholder(payload.get("answer"))


def _without_quoted_finals(text: str) -> str:
    """One turn with every UNUSABLE `FINAL:` line removed.

    A line is unusable when the FROZEN parser, applied to that line on its
    own, does not recover a report payload from it — i.e. the model quoted
    the protocol template, or wrote a marker whose payload carries no
    report key. Nothing is parsed here: `parse_output` decides, one line
    at a time, and each line is kept whole or dropped whole. A genuine
    FINAL elsewhere in the same turn survives untouched.
    """
    lines = text.split("\n")
    kept = []
    dropped = False
    for line in lines:
        if line.startswith(_FINAL_MARKER):
            parsed = parse_output(line)
            if parsed.kind != "final" or not _is_report_payload(parsed.final):
                dropped = True
                continue
        kept.append(line)
    return "\n".join(kept) if dropped else text


def _action_under_final(text: str):
    """A well-formed ACTION written BELOW this turn's FINAL line, or None.

    Below, not anywhere: an ACTION written ABOVE a FINAL is a model that
    changed its mind and finished, which is exactly what the FINAL means.
    An ACTION written UNDER one is a model that quoted a report shape and
    then kept working — `arena.model.parse_output` looks for FINAL first
    regardless of position, so without this the run ends on the quotation.
    """
    lines = text.split("\n")
    for index, line in enumerate(lines):
        if line.startswith(_FINAL_MARKER):
            below = parse_output("\n".join(lines[index + 1:]))
            return below if below.kind == "action" else None
    return None


@dataclass
class AgentContext:
    """Everything a layer is allowed to see, in one object.

    Passed to all six hooks as `ctx`. `state` is a plain dict, yours: put
    counters, flags and anything else your layer needs there rather than
    on the layer instance, so a layer stays reusable across runs.
    """

    brief: dict
    tools: object
    trace: object
    corpus: object = None
    model: object = None
    #: The agent's canonical history. Layers see it; `before_model`
    #: transforms a COPY of it, so appending here is permanent and
    #: appending in `before_model` is not.
    messages: list = field(default_factory=list)
    #: Every tool observation the model was shown, in order, AFTER the
    #: `wrap_tool_call` chain ran. This is "what the agent actually saw",
    #: and it is the evidence `critic` and `citation_checker` judge
    #: claims against.
    observations: list = field(default_factory=list)
    state: dict = field(default_factory=dict)
    step: int = 0
    stop_reason: str = ""

    @property
    def question(self) -> str:
        value = self.brief.get("question_vi")
        return value if isinstance(value, str) else ""

    @property
    def budget(self) -> dict:
        value = self.brief.get("budget")
        return value if isinstance(value, dict) else {}

    @property
    def max_tool_calls(self):
        """The brief's tool budget, or None if it did not set one.

        `arena.tools.Tools.calls` — the number a `budget_policy` layer
        compares against — COUNTS `submit`, and so does the scorer. A
        budget of 8 means seven useful calls plus the submit.
        """
        value = self.budget.get("max_tool_calls")
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            return None
        return value

    @property
    def observed_text(self) -> str:
        """Every observation, joined. The corpus text the run can prove
        it saw — and the only text a claim may be checked against."""
        return "\n".join(self.observations)

    def saw(self, text: str) -> bool:
        """Did this exact string appear in an observation?"""
        return bool(text) and text in self.observed_text


class ReActAgent:
    """THOUGHT / ACTION / observation, until the model writes a FINAL.

    Constructed with a model (`arena.model.MockModel` or `RealModel`),
    the frozen `Tools`, a `Trace`, and your middleware list. Everything
    else is keyword-only and has a working default.
    """

    def __init__(
        self,
        model,
        tools,
        trace,
        middleware: list | None = None,
        *,
        corpus=None,
        max_steps: int = MAX_STEPS,
        system_prompt: str = ARENA_SYSTEM_PROMPT,
        adaptive: bool | None = None,
    ) -> None:
        self.model = model
        self.tools = tools
        self.trace = trace
        self.middleware = MiddlewareStack(middleware)
        # A plain/logging-only scaffold remains reproducible. Installing a
        # final reviewer enables active repair; callers may opt out to measure
        # the same safety layers with the original passive loop.
        self.adaptive = self.middleware.has_final_review if adaptive is None else bool(adaptive)
        # The layers need the corpus to check a citation. `Tools` holds
        # one already, so a caller that does not pass one still works.
        self.corpus = corpus if corpus is not None else getattr(tools, "_corpus", None)
        self.max_steps = max(1, int(max_steps))
        self.system_prompt = system_prompt
        self.last_context: AgentContext | None = None
        # Per-run bookkeeping for the two `_parse` guards. Reset in
        # `run()`; kept on the agent rather than in `ctx.state`, which
        # belongs to the layers.
        self._final_deferrals = 0
        self._refused_final: dict | None = None

    # -- the run -------------------------------------------------------

    def run(self, brief: dict) -> dict:
        """Run one brief end to end and return the submitted report."""
        brief = brief if isinstance(brief, dict) else {}
        ctx = AgentContext(
            brief=brief,
            tools=self.tools,
            trace=self.trace,
            corpus=self.corpus,
            model=self.model,
        )
        self.last_context = ctx
        self._final_deferrals = 0
        self._refused_final = None

        self.trace.emit("agent_start", brief_id=str(brief.get("brief_id", "")))

        ctx.messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": ctx.question},
        ]
        self.middleware.before_agent(ctx)
        controller = RunController(ctx) if self.adaptive else None
        ctx.state["agent.controller"] = controller

        report: dict = {}
        model_turns = 0
        ctx.stop_reason = "max_steps"
        for step in range(self.max_steps):
            if controller and controller.best is not None and not controller.can_continue():
                report, ctx.stop_reason = controller.fallback(), "review_fallback"
                break

            ctx.step = step  # Count a turn only once it will call the model.

            outbound = controller.outbound(list(ctx.messages)) if controller else list(ctx.messages)
            outbound = self.middleware.before_model(ctx, outbound)
            response = self.middleware.wrap_model_call(ctx, self._call_model)(outbound)
            model_turns += 1
            response = self.middleware.after_model(ctx, response)
            if controller:
                controller.account(response)

            text = getattr(response, "text", None)
            if not isinstance(text, str):
                raise TypeError(
                    "the model (or a wrap_model_call/after_model hook) must return a "
                    f"ModelResponse whose .text is a str; got {type(text).__name__}"
                )

            parsed = self._parse(text)
            ctx.messages.append({"role": "assistant", "content": text})

            if parsed.kind == "final":
                report = parsed.final if isinstance(parsed.final, dict) else {}
                if controller:
                    issues = self.middleware.review_final(ctx, report)
                    retry_final = (step + 1 < self.max_steps
                                   and controller.assess(report, issues))
                    if retry_final:
                        recovery = controller.recovery_action()
                        if recovery:
                            name, args = recovery
                            controller.log("recovery_tool", tool=name)
                            observation = self._execute_tool(ctx, name, args)
                            ctx.observations.append(observation)
                            ctx.messages.append({"role": "user", "content": observation})
                        continue
                ctx.stop_reason = "final"
                break

            observation = self._observe(ctx, parsed)
            ctx.observations.append(observation)
            ctx.messages.append({"role": "user", "content": observation})

            if controller:
                continuation = controller.continuation_action()
                while continuation:
                    name, args = continuation
                    observation = self._execute_tool(ctx, name, args)
                    ctx.observations.append(observation)
                    ctx.messages.append({"role": "user", "content": observation})
                    continuation = controller.continuation_action()

            if controller and controller.forcing and controller.stalls >= controller.MAX_STALLS + 2:
                fallback = controller.fallback()
                report = fallback if fallback is not None else {}
                ctx.stop_reason = "review_fallback" if fallback is not None else "stalled_without_final"
                break

        if ctx.stop_reason == "max_steps" and controller:
            fallback = controller.fallback()
            if fallback is not None:
                report, ctx.stop_reason = fallback, "review_fallback"

        if ctx.stop_reason == "max_steps" and isinstance(self._refused_final, dict):
            # The loop ran out of steps and the only FINAL the model ever
            # wrote was one `_parse` put aside. Submit it: refusing bought
            # the model turns it did not use, and an empty report scores
            # zero, so this can only ever be an improvement.
            report = dict(self._refused_final)
            ctx.stop_reason = "refused_final"

        report = self.middleware.after_agent(ctx, report)
        # What gets submitted is what the layers returned — the scorer
        # reads the report off the `submit` event and refuses any claim
        # that is not in it (`NOT_SUBMITTED`).
        self.tools.submit(report)
        # No `elapsed_seconds` here on purpose: a wall clock inside the
        # harness would make the trace non-deterministic, and the frozen
        # runner stamps its own `agent_end` with the timing it measured.
        self.trace.emit("agent_end", stop_reason=ctx.stop_reason, steps=model_turns)
        return report

    # -- reading the model ---------------------------------------------

    def _parse(self, text: str):
        """Decode one model turn — with `arena.model.parse_output`, always.

        Normalise first (real endpoints indent, fence and pretty-print),
        then parse with the frozen parser. Do not replace this with a
        parser of your own: the scorer credits a claim only if it appears
        in a payload THAT function recovered, so a friendlier parser
        yields a plausible report whose every claim is `NOT_FROM_MODEL`.

        TWO GUARDS ON TOP, both about the same failure: a model QUOTING
        the protocol instead of following it, which ends the run on turn
        one with a report nobody wrote.

        1. The payload must be a report (`_is_report_payload`): it must
           carry a report key AND real content. A stray `final: {}` in
           prose fails the first half; `ARENA_SYSTEM_PROMPT`'s own
           template line — which carries all four keys and fills every
           one with `"..."` — fails the second. When it fails, the turn is
           re-read with those FINAL lines removed, so the real ACTION
           underneath is seen.
        2. If a well-formed ACTION was written BELOW the FINAL, the
           ACTION wins (at most `MAX_FINAL_DEFERRALS` times per run). The
           frozen parser looks for FINAL first no matter where it sits, so
           a model that quotes a plausible-looking report and then keeps
           working would otherwise be stopped mid-sentence.

        Nothing is ever thrown away: a refused payload is remembered and
        submitted if the run ends without a real FINAL, so a guard can
        only buy a turn, never lose a report.
        """
        parsed = parse_output(_canonicalise(text))
        if parsed.kind != "final":
            return parsed

        if _is_report_payload(parsed.final):
            action = _action_under_final(text)
            if action is None or self._final_deferrals >= MAX_FINAL_DEFERRALS:
                return parsed
            self._final_deferrals += 1
            self._refused_final = parsed.final
            return action

        if isinstance(parsed.final, dict) and any(
            key in parsed.final for key in REPORT_KEYS
        ):
            self._refused_final = parsed.final
        # Strict, NOT canonicalised: normalisation is what resurrects a
        # non-canonical marker such as `final: {}` in the first place, and
        # this path exists precisely to look underneath one.
        return parse_output(_without_quoted_finals(text))

    # -- the model -----------------------------------------------------

    def _call_model(self, messages: list[dict]):
        """The innermost model call — what `wrap_model_call` wraps.

        The `model_call` event is stamped HERE, from the response the
        model object returned, before any hook can see it. That ordering
        is the whole provenance story: `wrap_model_call` and `after_model`
        are student-owned and can return whatever they like, so a trace
        stamped from their return value would prove nothing at all.
        """
        response = self.model.complete(messages)
        # A frozen runner may take over `model_call` emission (it is the
        # only way to make the record unforgeable). It announces that by
        # setting `emits_model_call = True` on the model object.
        if not getattr(self.model, "emits_model_call", False):
            text = response.text if isinstance(response.text, str) else str(response.text)
            self.trace.emit(
                "model_call",
                prompt_tokens=response.prompt_tokens,
                completion_tokens=response.completion_tokens,
                # str is immutable — `Trace.emit` stores a reference to
                # whatever it is handed, so a mutable would let later code
                # rewrite history.
                output_text=text[:MAX_OUTPUT_TEXT_CHARS],
                step=self.last_context.step if self.last_context else 0,
            )
        return response

    # -- the tools -----------------------------------------------------

    def _observe(self, ctx: AgentContext, parsed) -> str:
        """Run one tool call through the `wrap_tool_call` chain and turn
        the result into the observation string the model is shown."""
        if parsed.kind != "action" or not parsed.tool:
            controller = ctx.state.get("agent.controller")
            if controller:
                controller.stalled("unreadable_action")
            # Not a THOUGHT/ACTION turn and not a FINAL either. Say so
            # rather than guessing — a real model that drifts off the
            # protocol needs to be told, and the mock never gets here.
            return (
                f"{TOOL_ERROR_PREFIX} không đọc được ACTION. Hãy trả lời đúng định dạng "
                "THOUGHT/ACTION hoặc THOUGHT/FINAL."
            )

        return self._execute_tool(ctx, parsed.tool, dict(parsed.args))

    def _execute_tool(self, ctx, name, args):
        """One genuine tool dispatch, or reuse of an already observed result."""
        controller = ctx.state.get("agent.controller")
        if name == "search":
            args = {**args, "query": _as_text(args.get("query")), "k": _as_k(args.get("k"))}
        elif name == "fetch_doc":
            args = {**args, "doc_id": _as_text(args.get("doc_id"))}
        if controller and controller.forcing:
            controller.stalled("ignored_finalize")
            return f"{TOOL_ERROR_PREFIX} Dừng công cụ: viết FINAL từ bằng chứng đã đọc."
        if controller:
            from harness.layers._resources import finalizing, tool_budget_spent
            if finalizing(ctx) or tool_budget_spent(ctx):
                controller.forcing = True
                return f"{TOOL_ERROR_PREFIX} Hết ngân sách truy xuất: viết FINAL ngay."
        result = controller.cached(name, args) if controller else None
        if result is None:
            call = self.middleware.wrap_tool_call(ctx, self._dispatch)
            result = call(name, args)
            if controller and result is not None and hasattr(result, "ok"):
                controller.result(name, args, result)
        if result is None or not hasattr(result, "ok"):
            return f"{TOOL_ERROR_PREFIX} layer trả về kết quả không hợp lệ cho {name}"
        if result.ok and not isinstance(result.content, str):
            return f"{TOOL_ERROR_PREFIX} nội dung công cụ phải là chuỗi"
        return result.content if result.ok else f"{TOOL_ERROR_PREFIX} {result.error}"

    def _dispatch(self, name: str, args: dict) -> ToolResult:
        """The innermost tool call — what `wrap_tool_call` wraps."""
        args = args if isinstance(args, dict) else {}
        if name == "search":
            return self.tools.search(_as_text(args.get("query")), k=_as_k(args.get("k")))
        if name == "fetch_doc":
            return self.tools.fetch_doc(_as_text(args.get("doc_id")))
        if name == "calc":
            return self.tools.calc(_as_text(args.get("expression")) or "0")
        return ToolResult(ok=False, content="", error=f"unknown tool: {name!r}")


def _as_text(value) -> str:
    return value if isinstance(value, str) else ("" if value is None else str(value))


def _as_k(value) -> int:
    try:
        k = int(value)
    except (TypeError, ValueError, OverflowError):
        return 5
    return max(1, min(MAX_SEARCH_K, k))


__all__ = [
    "AgentContext",
    "ReActAgent",
    "Middleware",
    "MAX_STEPS",
    "ARENA_SYSTEM_PROMPT_REAL",
    "REAL_MODEL_PROMPT_ADDENDUM",
    "real_model_system_prompt",
]
