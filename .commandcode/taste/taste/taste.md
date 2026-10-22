# Taste
- Prefers new implementations written in Rust. Confidence: 0.65
- Prefers provider-agnostic designs that wire up many external APIs/model providers rather than committing to a single vendor. Confidence: 0.6
- Cost-sensitive: wants to avoid recurring paid API/subscription fees, favoring self-hosted or local model options where they are viable. Confidence: 0.65
- Prefers explicit multi-tier agent architectures with separated, nameable roles (planner / executor / isolated evaluator) and enforced boundaries between them, rather than one monolithic agent loop. Confidence: 0.6
- Likes sandboxed, isolated execution for anything that runs generated code or tests. Confidence: 0.5
- Prefers a plan-first flow: review the plan once, then have the agent execute it end-to-end without re-confirming each step. Confidence: 0.6
- States quality expectations tersely ("make no mistakes") and expects the agent to close gaps with its own judgment rather than asking many clarifying questions. Confidence: 0.55
- Writes short, casual, typo-tolerant imperative messages; values brevity in responses. Confidence: 0.5
- When handing implementation work to OpenCode, prefers a repository `todo.md` plus a separate paste-ready handoff message. Confidence: 0.75
