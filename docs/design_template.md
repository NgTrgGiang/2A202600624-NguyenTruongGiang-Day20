# Design

## Problem

Trả lời một câu hỏi nghiên cứu dài (vd: "Research GraphRAG state-of-the-art and write a 500-word
summary"): cần (1) tìm nguồn, (2) phân tích/đối chiếu, (3) viết câu trả lời có trích dẫn cho đúng
đối tượng người đọc. Đầu vào là `ResearchQuery` (query, max_sources, audience); đầu ra là
`final_answer` kèm danh sách `sources` và trace của toàn bộ quá trình.

## Why multi-agent?

Single-agent làm tất cả trong một lượt: rẻ và nhanh, nhưng trộn lẫn việc tìm kiếm, phân tích và viết
nên khó kiểm soát chất lượng từng bước, dễ bịa nguồn và khó debug khi sai. Tách vai trò cho phép:
mỗi agent có một trách nhiệm rõ ràng, có điểm chèn guardrail (vd: critic kiểm tra grounding), và trace
chỉ ra chính xác bước nào tốn bao nhiêu / sai ở đâu. Đánh đổi: latency và cost cao hơn (xem benchmark).

## Agent roles

| Agent | Responsibility | Input | Output | Failure mode |
|---|---|---|---|---|
| Supervisor | Định tuyến bước kế tiếp + dừng (rule-based) | `ResearchState` | `next_route` | Loop vô hạn → chặn bằng `max_iterations` |
| Researcher | Tìm nguồn + viết research notes có `[n]` | query, sources | `sources`, `research_notes` | Search/LLM lỗi → ghi `errors`, set note tối thiểu |
| Analyst | Trích claims, đối chiếu, nêu bằng chứng yếu | `research_notes` | `analysis_notes` | LLM lỗi → fallback note, ghi `errors` |
| Writer | Tổng hợp câu trả lời cuối có trích dẫn | research+analysis notes | `final_answer` | LLM lỗi → fallback answer, ghi `errors` |
| Critic (bonus) | Kiểm tra citation coverage / grounding | `final_answer`, sources | review + cờ rủi ro | Coverage thấp → cảnh báo trong `errors` |

## Shared state

`ResearchState` ([core/state.py](../src/multi_agent_research_lab/core/state.py)) là single source of truth:

- `request` — câu hỏi gốc (không đổi suốt workflow).
- `iteration`, `route_history`, `next_route` — phục vụ định tuyến và guardrail; dễ debug đường đi.
- `sources`, `research_notes`, `analysis_notes`, `final_answer` — output bàn giao giữa các agent (handoff).
- `agent_results` — log mỗi lần agent chạy + metadata token/cost (để benchmark cộng dồn).
- `trace`, `errors` — quan sát & xử lý lỗi không làm sập workflow.

## Routing policy

Supervisor rule-based (deterministic, không tốn token), xem [supervisor.py](../src/multi_agent_research_lab/agents/supervisor.py):

```
iteration >= max_iterations            -> done (guardrail)
research_notes is None                 -> researcher
analysis_notes is None                 -> analyst
final_answer is None                   -> writer
use_critic và critic chưa chạy         -> critic
ngược lại                              -> done
```

LangGraph: `supervisor` là entry; conditional edge đọc `next_route` → worker hoặc `END`; mỗi worker
quay lại `supervisor` (xem [graph/workflow.py](../src/multi_agent_research_lab/graph/workflow.py)).

## Guardrails

- **Max iterations:** `Settings.max_iterations` (mặc định 6) — supervisor dừng khi vượt.
- **Timeout:** `Settings.timeout_seconds` truyền vào mỗi lời gọi LLM.
- **Retry:** `tenacity` (3 lần, backoff) trong `LLMClient.complete`.
- **Fallback:** mỗi agent bọc try/except → ghi `errors` và set output tối thiểu để pipeline tiến tiếp;
  graph có `recursion_limit` làm chốt chặn cuối.
- **Validation:** Pydantic schema cho mọi I/O; critic kiểm tra citation coverage và bật cờ rủi ro.

## Benchmark plan

Chạy `malab benchmark` trên các query trong [configs/lab_default.yaml](../configs/lab_default.yaml),
so sánh `baseline` vs `multi-agent`:

| Metric | Cách đo |
|---|---|
| Latency | wall-clock trong `run_benchmark` |
| Cost | cộng `cost_usd` từ `agent_results` metadata |
| Citation coverage | nguồn được trích / tổng nguồn ([utils/text.py](../src/multi_agent_research_lab/utils/text.py)) |
| Error rate | `len(state.errors)` |
| Quality | peer review 0-10 ([peer_review_rubric.md](peer_review_rubric.md)) |

Kết quả ghi ra `reports/benchmark_report.md` + `reports/trace_*.json`.
