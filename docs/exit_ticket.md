# Exit Ticket

Trả lời dựa trên benchmark thật (OpenAI) trong [reports/benchmark_report.md](../reports/benchmark_report.md):
baseline trung bình ~8.0s / $0.0004, multi-agent ~24.6s / $0.0012; citation coverage multi-agent = 1.0
ở cả 3 query, baseline = 0.0 ở 2/3 query.

## 1. Case nào nên dùng multi-agent? Vì sao?

Khi tác vụ **phức tạp, nhiều bước** và cần **độ tin cậy / khả năng kiểm soát chất lượng cao** — ví dụ
viết câu trả lời nghiên cứu **có trích dẫn**, cần kiểm chứng nguồn.

Lý do, dựa trên số liệu của chính hệ thống này:

- **Grounding tốt hơn rõ rệt:** multi-agent đạt citation coverage **1.0** ở mọi query, baseline chỉ
  **0.0** ở 2/3 query (gần như không trích nguồn). Tách Researcher → Analyst → Writer buộc mỗi bước
  bám vào nguồn, và Critic kiểm tra coverage trước khi kết thúc.
- **Dễ debug & kiểm soát:** mỗi agent một trách nhiệm, có `route_history` + trace từng bước, nên biết
  chính xác bước nào tốn bao nhiêu / sai ở đâu; có chỗ để cắm guardrail (max_iterations, critic).

## 2. Case nào KHÔNG nên dùng multi-agent? Vì sao?

Khi câu hỏi **đơn giản / một bước** và ưu tiên **rẻ và nhanh**.

- **Chi phí & độ trễ cao hơn ~3×:** multi-agent 24.6s vs 8.0s, $0.0012 vs $0.0004. Với tác vụ dễ,
  phần overhead điều phối + nhiều lời gọi LLM không đem lại đủ giá trị.
- **Phức tạp không cần thiết:** thêm agent là thêm điểm lỗi và state phải bàn giao; nếu single-agent
  đã đủ chất lượng (như query q3 baseline cũng đạt coverage 1.0) thì nên giữ đơn giản.

**Nguyên tắc rút ra:** chỉ thêm agent khi có lý do rõ ràng (một trách nhiệm mới, một guardrail mới);
đo bằng metric (coverage, cost, latency) thay vì cảm tính.
