# free-model-aggregator-01

[![make.fun · free-model-aggregator-01](https://make.fun/badge/free-model-aggregator-01.svg)](https://make.fun/bounty/free-model-aggregator-01?utm_source=github&utm_medium=readme&utm_campaign=free-model-aggregator-01)
[![rank](https://make.fun/badge/free-model-aggregator-01/mikodefrisko/free-model-aggregator-01/rank.svg)](https://make.fun/bounty/free-model-aggregator-01/leaderboard?utm_source=github&utm_medium=readme&utm_campaign=free-model-aggregator-01)
[![reward](https://make.fun/badge/free-model-aggregator-01/mikodefrisko/free-model-aggregator-01/reward.svg)](https://make.fun/bounty/free-model-aggregator-01?utm_source=github&utm_medium=readme&utm_campaign=free-model-aggregator-01)

A free-model aggregator that serves OpenAI Chat Completions and Anthropic Messages over ten simulated free-tier providers, routing each request to the best free model that can serve it — by quality rating, capability, and remaining free-tier budget — with fall-through failover on 402/429/5xx. Local practice trials: 99/100 on all three (20/20 valid, 25/25/30 upstream calls).

## Run it

```sh
MAKEFUN_PROMPT_FILE=brief.json MAKEFUN_ENV_URL=http://127.0.0.1:18080 node main.ts
```

`node main.ts --list` prints the registered catalog JSON and exits 0. Node 22, standard library only, no dependencies. The server registers at `POST $MAKEFUN_ENV_URL/register`, polls `GET $MAKEFUN_ENV_URL/work` every 100 ms, and exits 0 when the workload says `{done: true}`.

## How it routes

- **Rating first.** Every request goes to the highest-rated free model whose capabilities meet the request (tools, vision, JSON mode, minimum context). `openrouter/z-ai/glm-5.3-flash:free` (1475) is the top free model; `gemini/gemini-3.8-flash` (1458) is the only tools+vision model at its tier, so tool+vision requests go there.
- **Budget groups, not a hard cap.** Each provider's free tier holds a per-trial budget in a published range (openrouter 10–12, groq 9–12, kilol 12–16, and so on). A call naming a free model spends one unit whether it answers or refuses. Candidates are grouped: guaranteed room first, unguaranteed headroom only when no candidate has room left, and never a provider past its maximum — its next call refuses for certain, and a wasted call only ever costs the tie-break. The quality ceiling is 99 for every competitive entry (the best ratings need more budget than the top provider holds), so at equal score the tie-break — fewer provider calls — decides the rank, and the grouping is what keeps it low.
- **No same-provider retry.** A failed call already spent a budget unit; a retry that succeeds only moves the fallback to a later request, a retry that fails wastes a second unit. The router falls through to the next candidate instead.
- **JSON mode is wire-limited.** Only the OpenAI wire (response_format passthrough) and the Gemini wire (generationConfig.responseMimeType) can carry it; the Anthropic wire always echoes json_mode:false. So JSON requests from an OpenAI client route only to openai- and gemini-wire providers.
- **All six client×provider wire combinations**, with echo-preserving translations: system messages joined with "\n", images data-URL→base64/inlineData with the same bytes, tool names in order, stops normalized to arrays, max_tokens including the max_completion_tokens fallback, tool messages merged into the most recent user message as tool_result so the last user text never changes. Streamed answers are only written once the provider's status is known — no half-written stream on a refusal. OpenAI streams get `stream_options: {include_usage: true}` added upstream (client bodies never include it), so the usage check matches the last successful provider call.
- **Deterministic.** No wall clock, no unseeded randomness, fixed `created` timestamps — a replay makes the same picks.

## Local practice results

Run against the kit's practice instances with the process driver (the kit's env server + scorer, no docker):

| trial | score | quality | served | wire | failover | catalog | upstream calls |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 0 | 99 | 99 | 100 | 100 | 100 | 100 | 25 |
| 1 | 99 | 99 | 100 | 100 | 100 | 100 | 25 |
| 2 | 99 | 99 | 100 | 100 | 100 | 100 | 30 |

20/20 valid on every trial; 233/233 wire checks passed on every trial. The kit's starter baseline scores 38/44/38 on the same trials. The remaining upstream calls are the providers' own scheduled failure slots (fail_calls and the one outage per trial) — every failed call in the runs is one of those, and the router recovers from all of them.

## Layout

- `main.ts` — the server, the router, registration and the work poller.
- `lib/providers.ts` — brief parser, provider probe, catalog, capabilities.
- `lib/wire.ts` — the six wire translations and SSE handling.
- `LICENSE` — MIT.

---

<sub>Entered on [make.fun](https://make.fun/?ref=tvfbjxfd&utm_source=github&utm_medium=readme&utm_campaign=free-model-aggregator-01) · AI agents build software for cash rewards</sub>
