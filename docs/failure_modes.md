# Failure Modes & Fixes

Một số chế độ lỗi của hệ thống multi-agent này và cách đã/đang xử lý.

## 1. Supervisor loop vô hạn
**Triệu chứng:** một worker không tạo được output (vd research_notes vẫn `None`), supervisor cứ định
tuyến lại worker đó mãi.
**Nguyên nhân:** routing chỉ dựa trên "field còn trống hay chưa"; nếu worker fail im lặng thì field
không bao giờ được điền.
**Fix (đã làm):** (a) `max_iterations` chặn cứng số lần supervisor định tuyến; (b) mỗi worker bọc
try/except và **luôn set một giá trị tối thiểu** cho output của mình khi lỗi, để supervisor tiến bước;
(c) graph đặt `recursion_limit` làm chốt chặn cuối.

## 2. Hallucination / trích dẫn yếu
**Triệu chứng:** `final_answer` đưa ra khẳng định không có trong nguồn, hoặc gần như không trích `[n]`.
**Nguyên nhân:** LLM tự "bịa" khi nguồn mỏng; không có bước kiểm tra grounding.
**Fix (đã làm một phần):** CriticAgent tính `citation_coverage` và bật cờ rủi ro trong `state.errors`
khi coverage < ngưỡng. **Cải tiến tiếp:** để critic *chặn* và yêu cầu writer viết lại (loop có điều
kiện) thay vì chỉ cảnh báo; hoặc thêm LLM-judge chấm chất lượng tự động.

## 3. Lỗi/nghẽn từ nhà cung cấp (timeout, rate limit)
**Triệu chứng:** một lời gọi OpenAI/Tavily timeout hoặc 429 làm cả run hỏng.
**Nguyên nhân:** phụ thuộc mạng/quota.
**Fix (đã làm):** retry có backoff (`tenacity`) + `timeout_seconds` trong `LLMClient`; thiếu API key
→ tự fallback sang `MockLLMClient`/`MockSearchClient` để pipeline vẫn chạy được khi dev/test.

## 4. Mất context khi handoff
**Triệu chứng:** Writer thiếu thông tin Researcher đã có.
**Nguyên nhân:** state không mang đủ field cần thiết.
**Fix (đã làm):** mọi handoff đi qua `ResearchState` với các field tường minh (`sources`,
`research_notes`, `analysis_notes`) + `trace`/`agent_results` để truy vết; thêm field mới khi cần
thay vì truyền ngầm.
