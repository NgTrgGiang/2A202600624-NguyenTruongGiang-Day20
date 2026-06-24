# Benchmark Report

Single-agent baseline vs multi-agent. Quality is scored separately via peer review.

| Run | Latency (s) | Cost (USD) | Quality | Notes |
|---|---:|---:|---:|---|
| baseline-q1 | 10.26 | 0.0004 |  | steps=1, errors=0, citation_coverage=0.0, route=writer |
| multi-agent-q1 | 24.62 | 0.0014 |  | steps=4, errors=0, citation_coverage=1.0, route=researcher>analyst>writer>critic>done |
| baseline-q2 | 8.41 | 0.0003 |  | steps=1, errors=0, citation_coverage=0.0, route=writer |
| multi-agent-q2 | 31.84 | 0.0014 |  | steps=4, errors=0, citation_coverage=1.0, route=researcher>analyst>writer>critic>done |
| baseline-q3 | 5.33 | 0.0003 |  | steps=1, errors=0, citation_coverage=1.0, route=writer |
| multi-agent-q3 | 17.45 | 0.0010 |  | steps=4, errors=0, citation_coverage=1.0, route=researcher>analyst>writer>critic>done |

## Analysis

Across 3 queries (averages):

- Baseline: latency 8.00s, cost $0.0004
- Multi-agent: latency 24.64s, cost $0.0012

Multi-agent typically trades higher latency/cost for better grounding and separation of concerns; the baseline is cheaper but does everything in one pass. Add peer-review quality scores to complete the comparison.
