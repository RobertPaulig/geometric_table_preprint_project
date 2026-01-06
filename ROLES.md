# Roles and message protocol

This project uses three explicit personas:
- **ЧЕЛОВЕК** — the human operator who relays messages between the Architect and the Executor.
- **АРХИТЕКТОР** — defines tasks, methodology, and article text.
- **ИСПОЛНИТЕЛЬ** — edits code/TeX, runs experiments, commits/pushes, and reports results.

Message headers:
- Tasks from architect to executor: `АРХИТЕКТОР → ИСПОЛНИТЕЛЮ`.
- Explanations from architect to human: `АРХИТЕКТОР → ЧЕЛОВЕКУ`.
- Replies from executor back to architect: `ИСПОЛНИТЕЛЬ → АРХИТЕКТОРУ`.
- If the executor needs to speak directly to the human: `ИСПОЛНИТЕЛЬ → ЧЕЛОВЕКУ`.

Default response format for executor reports (short and structured):
1. Git status (hash, push, files).
2. Batch summary (counts, survival rate, K policy stats).
3. Regression results (beta, CI, p-value).
4. Matched-pairs results (counts, deltas, p-values, sanity flags).
5. Brief conclusion on whether the effect appears.
