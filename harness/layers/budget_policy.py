"""LỚP `budget_policy` — bài giảng Day 16, §3 (Budgets & Control Flow).

NHIỆM VỤ: kế hoạch của mô hình dài đúng 11 lượt gọi công cụ, bất kể brief
cho ngân sách bao nhiêu — và BỐN lượt cuối là rác có chủ ý: một lần search
lặp lại, một phép tính vô nghĩa, hai lần fetch lại tài liệu đã có trong
tay. Phần việc hữu ích nằm ở ĐẦU kế hoạch, nên cắt phần đuôi không mất một
điểm grounding nào mà lấy trọn phần điểm efficiency về tool call và token.

TÍN HIỆU:

    ctx.tools.calls >= ctx.max_tool_calls - reserve

CÁCH DỪNG: thêm `FINALIZE_SENTINEL` vào bên trong MỘT CÂU tiếng Việt bình
thường và đẩy vào cuối danh sách message trong `before_model`. `MockModel`
khoá theo token; một mô hình thật thì nghe câu tiếng Việt bao quanh nó.
Viết như vậy để cùng một lớp chạy được trên cả hai đường.

SENTINEL KHÔNG PHẢI TUỲ CHỌN — và không chỉ vì chuyện dừng.
`arena.model._first_user_content` lấy message user CUỐI CÙNG trước lượt
assistant đầu tiên làm câu hỏi của brief, và nó bỏ qua đúng những message
có mang `FINALIZE_SENTINEL`. Nếu bạn chèn một câu nhắc trơn không có
sentinel, mô hình sẽ đi search CHÍNH CÂU NHẮC ĐÓ: mọi brief truy xuất
cùng một mớ tài liệu vô can, mọi bậc thang điểm dịch chuyển đúng 0.00, và
không có một dòng lỗi nào báo cho bạn biết.

TRẢ VỀ `messages + [...]`, ĐỪNG `messages.append(...)`. Agent áp dụng
`before_model` lên một BẢN SAO của lịch sử, nên trả về danh sách mới nghĩa
là "nhắc trong đúng lượt này"; append vào chính danh sách được truyền vào
thì lời nhắc dính vĩnh viễn.

`reserve` KHÔNG PHẢI TRANG TRÍ: `Tools.calls` ĐẾM CẢ `submit`, và scorer
cũng đếm như vậy. Brief cho `max_tool_calls: 8` nghĩa là bảy lượt hữu ích
cộng một lượt submit. Dừng ở `calls >= 8` là tiêu lố đúng một lượt, lần
nào cũng lố.

MỘT HOOK LÀ CHƯA ĐỦ — ĐÃ ĐO. `before_model` chỉ chặn được khi mỗi lượt
model tiêu đúng MỘT lượt công cụ. Không phải vậy: lớp `retry` (§7) có thể
tiêu ba lượt trong CÙNG một vòng, nên một vòng bắt đầu khi còn thiếu đúng
một lượt vẫn kết thúc ở trên ngưỡng. Đo trên full stack: 34/120 lượt chạy
kết thúc ở 9+ lượt gọi trong khi brief cho 8, efficiency 12.06 thay vì
14.24 — trong khi `budget_policy` chạy MỘT MÌNH thì sạch cả 120 lượt.
Vì thế lớp này có thêm `wrap_tool_call`: khi ngân sách chỉ còn phần dự
trữ, TỪ CHỐI gọi công cụ (trả về `ToolResult(ok=False, ...)`, đừng raise —
agent phải sống để còn chốt FINAL). Nửa còn lại nằm ở `retry`: hook
`wrap_tool_call` của `budget_policy` bọc NGOÀI vòng lặp thử lại nên không
nhìn thấy các lượt gọi lại; chỉ chính `retry` mới chặn được `retry`.

CẢNH BÁO ĐÃ ĐO ĐƯỢC — ĐỪNG NÉN NGỮ CẢNH Ở ĐÂY. `before_model` trông rất
hợp lý để "tóm tắt cho gọn", nhưng `MockModel` chỉ trích được câu nào
xuất hiện NGUYÊN VĂN trong danh sách message NÓ ĐANG NHẬN. Một lớp nén
ngữ cảnh tử tế làm mô hình mất khả năng trích dẫn chính những tài liệu nó
vừa đọc: -47.16 điểm trên full stack (92.52 -> 45.36), không có một
thông báo lỗi nào.

CÔNG CỤ CÓ SẴN:
    from arena.model import FINALIZE_SENTINEL
    from arena.tools import ToolResult
    ctx.tools.calls      -> số lượt gọi công cụ đã dùng (kể cả submit)
    ctx.max_tool_calls   -> ngân sách của brief, hoặc None nếu brief không đặt

Cài đặt:  ReActAgent(..., middleware=[..., BudgetPolicy(), ...])
Xem `harness/middleware.py` để biết thứ tự các hook.
"""

from __future__ import annotations

import time

from arena.model import FINALIZE_SENTINEL
from arena.tools import ToolResult

from harness.middleware import Middleware
from harness.layers._resources import finite_limit, tool_budget_spent

#: Dành lại cho lượt `submit` mà agent vẫn còn phải gọi.
DEFAULT_RESERVE = 1

NUDGE = (
    "Dành ngân sách còn lại để kết luận. Hãy trả lời ngay bằng bằng chứng đang có, "
    f"không gọi thêm công cụ nào nữa. {FINALIZE_SENTINEL}"
)


class BudgetPolicy(Middleware):
    """Ép mô hình chốt FINAL ngay khi ngân sách công cụ đã tiêu hết."""

    name = "budget_policy"

    def __init__(self, reserve: int = DEFAULT_RESERVE, *, reserve_tokens=512, clock=None) -> None:
        self.reserve = max(0, int(reserve))
        self.reserve_tokens = max(0, int(reserve_tokens))
        self.clock = clock if clock is not None else time.monotonic

    def before_agent(self, ctx):
        started = self.clock()
        seconds = finite_limit(ctx.budget.get("max_seconds"))
        ctx.state["budget.started"] = started
        ctx.state["budget.clock"] = self.clock
        ctx.state["budget.deadline"] = (started + seconds - min(1.0, seconds * 0.1)
                                        if seconds is not None else None)
        ctx.state["budget.reserve"] = self.reserve
        ctx.state["budget.reserve_tokens"] = self.reserve_tokens
        ctx.state["budget.tokens"] = 0
        ctx.state["budget.prompt_chars"] = 0
        ctx.state["budget.prompt_tokens"] = 0
        ctx.state["budget.finalizing"] = False

    def _reason(self, ctx):
        if ctx.state.get("budget.finalizing"):
            return ctx.state.get("budget.reason", "tool_calls")
        if tool_budget_spent(ctx, self.reserve):
            return "tool_calls"
        limit = finite_limit(ctx.budget.get("max_tokens"))
        if limit is not None and ctx.state.get("budget.tokens", 0) >= limit - self.reserve_tokens:
            return "tokens"
        seconds = finite_limit(ctx.budget.get("max_seconds"))
        started = ctx.state.get("budget.started")
        if seconds is not None and started is not None:
            if self.clock() - started >= seconds - min(1.0, seconds * 0.1):
                return "time"
        return ""

    def _spent(self, ctx) -> bool:
        return bool(self._reason(ctx))

    def before_model(self, ctx, messages):
        reason = self._reason(ctx)
        chars = sum(len(m.get("content", "")) for m in messages
                    if isinstance(m, dict) and isinstance(m.get("content"), str))
        previous_chars = ctx.state.get("budget.prompt_chars", 0)
        previous_tokens = ctx.state.get("budget.prompt_tokens", 0)
        limit = finite_limit(ctx.budget.get("max_tokens"))
        if not reason and limit is not None and previous_chars and previous_tokens:
            # Calibrate to actual endpoint usage (Vietnamese differs from
            # chars/4). Preserve the full evidence; finalize before a new
            # retrieval would leave too little for the following FINAL.
            estimate = max(previous_tokens, int(chars * previous_tokens / previous_chars)) + 128
            if ctx.state.get("budget.tokens", 0) + 2 * estimate + self.reserve_tokens > limit:
                reason = "tokens_forecast"
        ctx.state["budget.prompt_chars"] = chars
        if not reason:
            return messages
        ctx.state["budget.finalizing"] = True
        ctx.state["budget.reason"] = reason
        if any(FINALIZE_SENTINEL in m.get("content", "")
               for m in messages if isinstance(m, dict) and isinstance(m.get("content"), str)):
            return messages
        # Keep every observation intact; a one-turn nudge, no fabricated FINAL.
        return messages + [{"role": "user", "content": NUDGE}]

    def after_model(self, ctx, response):
        ctx.state["budget.prompt_tokens"] = finite_limit(getattr(response, "prompt_tokens", None)) or 0
        usage = sum(finite_limit(getattr(response, name, None)) or 0
                    for name in ("prompt_tokens", "completion_tokens"))
        ctx.state["budget.tokens"] = ctx.state.get("budget.tokens", 0) + usage
        return response

    def wrap_tool_call(self, ctx, call, name, args):
        reason = self._reason(ctx)
        if reason:
            ctx.state["budget.finalizing"] = True
            ctx.state["budget.reason"] = reason
            ctx.state["budget.blocked"] = ctx.state.get("budget.blocked", 0) + 1
            return ToolResult(ok=False, content="", error=f"Hết ngân sách ({reason}); hãy chốt FINAL.")
        return call(name, args)
