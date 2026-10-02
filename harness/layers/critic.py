"""LỚP `critic` — bài giảng Day 16, §2 (Reflection & Self-Critique).

NHIỆM VỤ: mô hình KHÔNG BAO GIỜ nói "tôi không biết". `abstain` bị gán
cứng `False`, và nó bịa theo ba kiểu khác nhau:

  (a) brief `absent`  -> bịa ra một con số không có trong tài liệu nào.
  (b) không có bằng chứng -> bịa ra một câu chung chung vô thưởng vô phạt.
  (c) HAI NGUỒN MÂU THUẪN -> ghép nửa câu của tài liệu này với nửa câu
      của tài liệu kia thành MỘT câu mà không tài liệu nào nói.

TÍN HIỆU (chỉ một dòng): câu trong `claim["text"]` có xuất hiện NGUYÊN VĂN
trong bằng chứng agent đã thực sự đọc hay không —

    text in ctx.observed_text

Trên một brief có bằng chứng tốt thì mọi claim đều thoả điều kiện này,
nên critic xây trên tín hiệu đó không báo động giả.

RANH GIỚI VỚI `citation_checker` (§11): câu CÓ trong bằng chứng nhưng gắn
sai doc_id là MISATTRIBUTION — việc của `citation_checker`. Câu KHÔNG có
trong bất kỳ bằng chứng nào là FABRICATION — việc của bạn ở đây. Hai điều
kiện loại trừ nhau, đừng làm phần việc của lớp kia.

ĐIỂM SỐ (đọc kỹ, đây là nơi kiếm nhiều điểm nhất):
  * Một claim bịa bị chấm `HALLUCINATED`: mất điểm precision VÀ mất trọn
    15 điểm honesty, trên MỌI brief.
  * Trên brief `is_absent`, `abstain: true` được 0.75 recall + trọn 15
    điểm honesty. "Không có số liệu" CHÍNH LÀ câu trả lời đúng.
  * Trên brief mâu thuẫn, ĐỪNG trông đợi "nêu cả hai phía" tự động cho
    recall đầy đủ: recall chấm THEO TỪNG required_fact bằng key terms
    của chính fact đó, không phải theo số vế đã trích dẫn — nếu nửa câu
    mô hình thực sự viết ra không phủ hết từ khoá của một fact (mô hình
    ghép câu ở chỗ NÓ chọn, không nhất thiết đúng ranh giới required_fact),
    fact đó vẫn 0 điểm dù trích dẫn đúng. Trên `pub-04-lam-viec-tu-xa` cụ
    thể, recall từ FINAL ghép cố định của MockModel chỉ đạt 0.5 —
    đo được, không phải suy đoán. Vẫn nên làm: `abstain: true` sau khi nêu
    cả hai phía được 0.5 recall + trọn 15 điểm honesty, và điểm recall lấy
    theo `max(...)` nên làm cả hai không bao giờ THIỆT — chỉ đừng trông
    đợi nó vượt sàn 0.5 trên brief này.
  * Xoá claim là hợp lệ. SỬA CHỮ trong `claim["text"]` thì KHÔNG: thêm
    một dấu chấm cuối câu cũng đủ làm claim mất cả provenance lẫn hỗ trợ
    (đo được: -40 điểm). Chỉ được xoá, giữ nguyên, hoặc cắt bớt.

GỢI Ý cho trường hợp (c): câu bị ghép là hai đoạn DO CHÍNH MÔ HÌNH viết,
dán với nhau bằng một liên từ (" và "). Cắt đúng chỗ dán thì hai nửa vẫn
là chữ của mô hình — vẫn qua được kiểm tra provenance. Muốn biết cắt đúng
chưa: cả hai nửa phải xuất hiện nguyên văn trong `ctx.observed_text` và
phải thuộc HAI tài liệu khác nhau. Cắt sai thì một nửa sẽ vắt qua hai tài
liệu và không quan sát nào chứa nó.

CÔNG CỤ CÓ SẴN:
    ctx.observed_text  -> toàn bộ quan sát agent đã thấy, nối lại
    ctx.saw(text)      -> text có trong quan sát không
    ctx.corpus.docs    -> danh sách Doc (doc_id, title, body); qua
                          `ctx.corpus`, `Doc.tags` LUÔN RỖNG — CẢ Ở VÒNG
                          LUYỆN TẬP LẪN VÒNG CHẤM ĐIỂM, vì corpus mà code
                          của bạn cầm bị gỡ nhãn bẫy ('outdated',
                          'contradiction', 'injection'…) ngay khi runner
                          dựng lên nó, không phải chỉ lúc chấm điểm. Đọc
                          nhãn là tra bảng chứ không phải kỹ năng lab này
                          chấm. Ở vòng LUYỆN TẬP seed 42 thì file TRÊN ĐĨA
                          `data/corpus/*.json` (khác với `ctx.corpus`)
                          vẫn có nhãn: hard-code được từ đó, và điều đó
                          được nói thẳng ra ở đây thay vì giấu đi.
    ctx.state          -> dict tuỳ bạn dùng để ghi số liệu gỡ lỗi

Cài đặt:  ReActAgent(..., middleware=[InjectionGuard(), Critic(), ...])
Xem `harness/middleware.py` để biết thứ tự các hook.
"""

from __future__ import annotations

import re

from harness.middleware import FinalIssue, Middleware
from harness.layers._evidence import (
    EvidenceIndex, MAX_CLAIMS, MAX_INPUT_CLAIMS, MAX_PER_DOC,
    MAX_QUOTE_CHARS, citations, conflict, content_words, remember_result,
)

_CONNECTORS = re.compile(r" và | nhưng | trong khi | whereas | but | and |; ", re.IGNORECASE)
_NO_EVIDENCE = "Không đủ bằng chứng đã kiểm chứng để trả lời."
_QUALIFICATION = re.compile(
    r"\b(?:phải|chỉ khi|trừ khi|ngoại lệ|áp dụng|phiên bản|phạm vi|"
    r"must|only if|unless|except|applicable|effective|version)\b", re.IGNORECASE)
_STATISTICS_UNIT = re.compile(
    r"\b(?:phòng|bộ phận|bên|ban|đội|nhóm)\s+([^.;?\n]{2,50}?)\s+"
    r"(?:giữ|cung cấp|ghi nhận|lưu|quản lý)\s+(?:thống kê|số liệu|báo cáo)", re.IGNORECASE)


class Critic(Middleware):
    """Xoá những gì bằng chứng không đỡ; abstain khi không còn gì."""

    name = "critic"

    def review_final(self, ctx, report):
        """Diagnose before deleting: let a real model recover missing facts.

        Read-only, no after_agent calls and no answer-key inspection. The
        eventual submission still gets the conservative safety filter.
        """
        issues = []
        raw = report.get("claims")
        if (not isinstance(raw, list) or not isinstance(report.get("abstain"), bool)
                or not isinstance(report.get("answer"), str)
                or not isinstance(report.get("citations"), list)):
            issues.append(FinalIssue("schema", "Sửa kiểu dữ liệu: answer chuỗi, claims/citations mảng, abstain boolean."))
        raw = raw[:MAX_INPUT_CLAIMS] if isinstance(raw, list) else []
        index = EvidenceIndex(ctx)
        supported, invalid, incomplete, stale = [], [], [], False
        for position, claim in enumerate(raw):
            source = index.source_for(claim.get("text"), claim.get("doc_id")) if isinstance(claim, dict) else None
            if not source:
                invalid.append(position + 1)
            else:
                supported.append(claim)
                stale |= source.superseded
                text = claim["text"]
                # Diagnose only a complete line already visible to the model.
                # Never supply corpus text as replacement model-authored claims.
                for line_number, line in enumerate(source.lines, 1):
                    if (text in line and len(line) <= MAX_QUOTE_CHARS
                            and source.supports(line)):
                        before, _, after = line.partition(text)
                        if _QUALIFICATION.search(before + " " + after):
                            incomplete.append((position + 1, source.doc_id, line_number))
                        break
        if invalid:
            issues.append(FinalIssue("unsupported_claim", "Claims " + ",".join(map(str, invalid[:10]))
                                     + " không phải đoạn nguyên văn một dòng đã đọc. Đọc lại bằng chứng và chép đúng; không ghép nguồn."))
        if not raw and report.get("abstain") is not True:
            issues.append(FinalIssue("empty_claims", "Chưa có claim hỗ trợ. Tìm/đọc nguồn liên quan hoặc abstain nếu thực sự thiếu dữ liệu."))
        if incomplete:
            locations = "; ".join(f"claim {p}: dòng {n} trong toàn văn {doc_id}"
                                  for p, doc_id, n in incomplete[:10])
            issues.append(FinalIssue("incomplete_quote", locations
                                     + ". Claim đang cắt một dòng nguồn thành câu rời. Chép TRỌN riêng dòng vật lý đó, kể cả các câu tiếp theo trên CÙNG dòng. Không ghép tiêu đề/dòng hiệu lực/dòng khác; tối đa 500 ký tự, giữ nguyên chữ."))
        unit = _STATISTICS_UNIT.search(ctx.question)
        if unit and supported:
            wanted = " ".join(unit.group(1).casefold().split())
            if not any(wanted in " ".join(c["text"].casefold().split()) for c in supported):
                issues.append(FinalIssue("statistics_unit", "Số liệu chưa thuộc bộ phận được hỏi: "
                                         + unit.group(1) + ". Tìm báo cáo thống kê của đúng bộ phận và chủ đề nghiệp vụ; không dùng con số của đơn vị khác."))
        if stale and report.get("abstain") is not True:
            issues.append(FinalIssue("superseded_source", "Có nguồn đã hết hiệu lực/bị thay thế. Tìm phiên bản hiện hành và kiểm tra phạm vi áp dụng."))
        contradictory = any(index.comparable(a, b) and conflict(a["text"], b["text"])
                            for i, a in enumerate(supported[:MAX_CLAIMS])
                            for b in supported[i + 1:MAX_CLAIMS])
        if contradictory and report.get("abstain") is not True:
            issues.append(FinalIssue("conflict", "Các nguồn có số liệu mâu thuẫn. Kiểm tra hiệu lực/phạm vi; chưa giải quyết được thì abstain và nêu hai nguồn."))
        if re.search(r"\(a\).+\(b\)", ctx.question, re.IGNORECASE | re.DOTALL):
            verdict = report.get("verdict")
            if not isinstance(verdict, str) or not verdict.strip():
                issues.append(FinalIssue("missing_verdict", "Câu hỏi có phương án. Thêm verdict chép đúng một phương án từ câu hỏi, dựa trên các claim đã đọc."))
            elif (len(verdict.strip()) < 6
                  or " ".join(verdict.casefold().split()) not in " ".join(ctx.question.casefold().split())):
                issues.append(FinalIssue("invalid_verdict", "verdict chưa phải nguyên văn phương án từ câu hỏi; không chỉ ghi chữ cái hay mã tự đặt."))
        return tuple(issues)

    def wrap_tool_call(self, ctx, call, name, args):
        result = call(name, args)
        remember_result(ctx, name, args, result)
        return result

    def _split(self, text, index):
        # Examine only a bounded number of candidate boundaries. Both
        # halves remain literal substrings of a model-authored quotation.
        for number, match in enumerate(_CONNECTORS.finditer(text)):
            if number >= 32:
                break
            left, right = text[:match.start()].strip(), text[match.end():].strip()
            a, b = index.source_for(left), index.source_for(right)
            if a and b and a.doc_id != b.doc_id:
                pieces = [{"text": left, "doc_id": a.doc_id}, {"text": right, "doc_id": b.doc_id}]
                return pieces, index.comparable(*pieces) and conflict(left, right, match.group())
        return [], False

    def after_agent(self, ctx, report):
        model_abstained = report.get("abstain") is True
        raw = report.get("claims")
        raw = raw[:MAX_INPUT_CLAIMS] if isinstance(raw, list) else []
        index = EvidenceIndex(ctx)
        kept, contradictory, changed = [], False, False
        for claim in raw:
            if not isinstance(claim, dict) or not isinstance(claim.get("text"), str):
                changed = True
                continue
            text = claim["text"]
            if len(text) > 16_000:  # No unbounded splitting of hostile output.
                changed = True
                continue
            source = index.source_for(text, claim.get("doc_id"))
            if source:
                # Leave attribution to CitationChecker, including when the
                # critic is run alone for ablation measurements.
                kept.append(dict(claim))
                continue
            trimmed = text.strip()
            wrappers = (("\"", "\""), ("“", "”"), ("'", "'"), ("`", "`"))
            for opening, closing in wrappers:
                if trimmed.startswith(opening) and trimmed.endswith(closing) and len(trimmed) > 2:
                    trimmed = trimmed[1:-1].strip()
                    break
            source = index.source_for(trimmed, claim.get("doc_id"))
            if source:
                kept.append({**claim, "text": trimmed, "doc_id": source.doc_id})
                changed = True
                continue
            pieces, has_conflict = self._split(text, index)
            kept.extend(pieces)
            contradictory |= has_conflict
            changed = True

        # Independent quotes can also disagree, even without a fused claim.
        for i, left in enumerate(kept[:MAX_CLAIMS]):
            for right in kept[i + 1:MAX_CLAIMS]:
                if index.comparable(left, right):
                    contradictory |= conflict(left["text"], right["text"])

        # Only rank if the model supplied too much. Never consult required
        # facts, gold verdicts, tags or brief IDs to choose evidence.
        question_words = content_words(ctx.question)
        if len(kept) > MAX_CLAIMS:
            kept.sort(key=lambda c: -len(content_words(c["text"]) & question_words))
        selected, seen, counts = [], set(), {}
        for claim in kept:
            text, doc_id = claim["text"], claim.get("doc_id")
            if len(text) > MAX_QUOTE_CHARS:
                text = text[:MAX_QUOTE_CHARS]  # A substring, never repaired text.
                claim = {**claim, "text": text}
                changed = True
            key = (text, doc_id if isinstance(doc_id, str) else "")
            if key in seen or counts.get(key[1], 0) >= MAX_PER_DOC or len(selected) >= MAX_CLAIMS:
                changed = True
                continue
            seen.add(key)
            counts[key[1]] = counts.get(key[1], 0) + 1
            selected.append(claim)

        report = {**report, "claims": selected, "citations": citations(selected)}
        abstain = report.get("abstain")
        if not isinstance(abstain, bool):
            report["abstain"] = isinstance(abstain, str) and abstain.strip().casefold() == "true"
        if not isinstance(report.get("answer"), str) or not report["answer"].strip():
            report["answer"] = "Bằng chứng đã kiểm chứng: " + " ".join(c["text"] for c in selected)
        if not selected:
            report.update(abstain=True, answer=_NO_EVIDENCE)
            report.pop("verdict", None)  # No evidence, no confident conclusion.
        elif contradictory:
            # A model can legitimately choose the question's "cannot
            # determine" option while abstaining. Keep that model-authored
            # decision when it explicitly abstained and quoted an offered
            # option; do not discard every synthesis verdict by default.
            verdict = report.get("verdict")
            model_choice = (model_abstained and isinstance(verdict, str)
                            and len(verdict.strip()) >= 6
                            and re.search(r"\(a\).+\(b\)", ctx.question, re.IGNORECASE | re.DOTALL)
                            and " ".join(verdict.casefold().split()) in " ".join(ctx.question.casefold().split()))
            report.update(abstain=True, answer="Các nguồn đã đọc có thông tin mâu thuẫn: "
                          + " ".join(c["text"] for c in selected))
            if not model_choice:
                report.pop("verdict", None)
        elif changed:
            # Do not leave removed hallucinations in the narrative answer.
            report["answer"] = "Bằng chứng đã kiểm chứng: " + " ".join(c["text"] for c in selected)
        ctx.state["critic.kept"] = len(selected)
        ctx.state["critic.removed"] = max(0, len(raw) - len(selected))
        ctx.state["critic.conflict"] = contradictory
        return report
