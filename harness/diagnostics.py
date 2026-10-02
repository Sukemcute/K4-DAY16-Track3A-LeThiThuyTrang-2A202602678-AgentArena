"""Summarise recorded runs; diagnose failures without changing scored output."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path


def diagnose(row):
    issues = []
    flags = row.get("flags", [])
    models, tools = row.get("model_calls", 0), row.get("tool_calls", 0)
    if models == 1 and tools <= 1:
        issues.append("premature_final_without_retrieval")
    if "no_final_output" in flags or row.get("final_outputs") == 0:
        issues.append("no_decodable_model_final")
    if row.get("error"):
        issues.append("runtime_error")
    detail = row.get("detail", {})
    grounding = detail.get("grounding", {})
    counts = grounding.get("verdict_counts", {})
    issues.extend(label.lower() for label in ("HALLUCINATED", "MISATTRIBUTED", "UNRETRIEVED", "NOT_FROM_MODEL")
                  if counts.get(label, 0))
    if grounding.get("recall", 1) < 1:
        issues.append("incomplete_fact_or_verdict_coverage")
    if grounding.get("recall", 1) == 0 and counts.get("SUPPORTED", 0):
        issues.append("supported_quotes_do_not_answer_question")
    transcript = row.get("trace_jsonl", "")
    events = [json.loads(line) for line in transcript.splitlines() if line.strip()]
    repairs = [e.get("issues", "") for e in events if e.get("hook") == "review_rejected"]
    return {"issues": issues, "repairs": repairs,
            "cache_hits": sum(e.get("hook") == "cache_hit" for e in events),
            "recovery_tools": sum(e.get("hook") == "recovery_tool" for e in events)}


def render(paths):
    lines = ["# Phân tích lượt chạy", "",
             "Chẩn đoán từ kết quả/trace đã lưu. Đây là tín hiệu để điều tra, không phải kết luận về private test hoặc điểm model thật.", ""]
    for path in paths:
        payload = json.loads(path.read_text(encoding="utf-8"))
        rows = payload.get("runs", [])
        lines.extend([f"## {path.name}", "",
                      "| Brief / tình huống | Tổng | Model / tool | Chẩn đoán | Repair / cache |",
                      "|---|---:|---:|---|---:|"])
        totals = Counter()
        for row in rows:
            diagnostic = diagnose(row)
            totals.update(diagnostic["issues"])
            label = row.get("brief_id", row.get("mode", "unknown"))
            if "adaptive" in row:
                label += " / " + ("adaptive" if row["adaptive"] else "passive")
            score = row.get("total", 0)
            lines.append(f"| {label} | {score:.2f} | {row.get('model_calls', 0)} / {row.get('tool_calls', 0)} | "
                         + (", ".join(diagnostic["issues"]) or "Không có lỗi thuộc các mẫu đang kiểm tra")
                         + f" | {len(diagnostic['repairs'])} / {diagnostic['cache_hits']} |")
        lines.extend(["", "Tần suất: " + (", ".join(f"{k}={v}" for k, v in totals.items()) or "0 mẫu lỗi"), ""])
    lines.extend(["## Hướng cải thiện từ lỗi", "",
                  "- FINAL trước truy xuất: bootstrap search thật, cho model đọc nguồn và viết lại FINAL.",
                  "- Quote không nguyên văn: phản hồi vị trí claim lỗi cho model sửa trước khi filter; giữ kiểm chứng dòng và provenance.",
                  "- Quote hợp lệ nhưng recall thấp: đổi truy vấn theo thuật ngữ của tài liệu, kiểm tra từng ý hỏi; sửa citation đơn thuần không giải quyết được.",
                  "- Synthesis thiếu verdict: yêu cầu model quyết định một phương án bằng claim đã đọc.",
                  "- Lặp công cụ: cache payload thành công đã qua guard; dừng vòng lặp không tạo bằng chứng mới.",
                  "- Thiếu FINAL khi hết tài nguyên: giữ FINAL trước đó của model để fallback; không tạo model_call hoặc FINAL giả.", ""])
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("inputs", type=Path, nargs="+")
    parser.add_argument("--out", type=Path, default=Path(".agent/FAILURE_ANALYSIS.md"))
    args = parser.parse_args()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(render(args.inputs), encoding="utf-8")
    print(args.out)


if __name__ == "__main__":
    main()
