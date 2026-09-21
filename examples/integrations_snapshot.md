# langchain-typesafe 0.0.1a2 (PyPI) — H7 snapshot

> Source: https://pypi.org/project/langchain-typesafe/ — fetched 2026-09-18. Uploaded 2026-09-17, trusted publishing, MIT-style LangChain integration.

## TypeSafeClassifier — constructor form (differs from blog's .invoke-time questions)

```python
from langchain_typesafe import Choice, Noul, Score, TypeSafeClassifier
classifier = TypeSafeClassifier(questions={
    "department": Choice(instructions="Which team should handle this?",
                         criteria={"billing": "...", "technical": "..."}),
    "urgent": Noul(instructions="Does this message express urgency?"),
    "frustration": Score(instructions="...", criteria=["calm", "frustrated", "angry"]),
})
result = classifier.invoke("Stripe has failed to connect for three days. Help ASAP.")
# result.choices["department"].choice / result.nouls["urgent"].noul / result.scores[...].score
```

Runnable semantics: ainvoke, batching, callbacks, tracing. LangChain BaseMessage sequences allowed anywhere in state → recursively converted to {role, content}.

## AutoModeMiddleware — the public tool-risk Noul definition

```python
from langchain_typesafe import NoulCriteria
from langchain_typesafe.experimental.middleware import AutoModeMiddleware
auto_mode = AutoModeMiddleware(
    tools=[delete_file],
    criteria=NoulCriteria(
        true="The call writes, deletes, publishes, or changes access.",
        false="The call only reads public or user-provided data.",
    ),
)
agent = create_agent(model, tools=[read_file, delete_file], middleware=[auto_mode])
```

"Configured calls whose risk probability meets or exceeds the threshold return an error ToolMessage." → **a production Noul question over a tool call; label derivable from tool-call outcomes (H7 dataset candidate).**

## ModelRouterMiddleware

Choice over {fast: "Simple, well-scoped tasks.", powerful: "Complex tasks requiring deeper reasoning."} — "Choose the least costly model suited to the task." Classified once per run from latest human message; full ChoiceAnswer stored in agent state.

## Error surface

TypeSafeRateLimitError etc. map onto LangChain ModelAuthenticationError/ModelRateLimitError; TypeSafeAPIError exposes status, parsed body, headers, sanitized endpoint, request id.

# jev-mcp by rashedInt32 (glama) — H7 snapshot

> Source: https://glama.ai/mcp/servers/rashedInt32/jev-mcp — fetched 2026-09-18. MIT. Node 20.12+.

- Tools: `jev_classify` (Choice), `jev_score` (Score), `jev_check` (Noul), `jev_ask` (batch all three), `jev_models`.
- Every answer validated against the question sent; distribution always returned; adds `none` no-match option by default (disable with add_none: false).
- Confidence gating built in: act / review / abstain from `act_above` (default 0.8) and `review_above` (default 0.5); jev_check yes/no/uncertain at 0.7/0.3. README: "These defaults are starting points... Calibrate them on your own data" + "Confidence describes how concentrated the distribution is. It is not a claim that the answer is correct."
- 255 options = API hard cap; >255 → two-pass (window pick + rank within).
- Batched call economics (their measurement, doc cookbook numbers): 12.2x cheaper, 10.0x faster vs one call per question, no change in answers.
- latency_ms returned with every judgment — cost+confidence pairing by design.
- Design maxims worth mirroring: caller owns option set; probabilities always returned; nothing silently dropped/truncated; only JSON-RPC on stdout.

# vinnylarouge/jevlike — H7 snapshot

> Source: https://github.com/vinnylarouge/jevlike — fetched 2026-09-18. MIT.

- Independent open-source "Jev-like" model: text + N options → one probability per option, single pass. "TypeSafe has not published its design. This repository is an independent starter model with the same input and output shape."
- Architecture: per-option query vector → attention over context tokens → per-option context vector → shared dot product → softmax across options. Encoder: byte embeddings from scratch OR frozen HF encoder (example: Qwen/Qwen2.5-0.5B, rank-256 scorer head).
- **jevlike-eval prints top-1/top-3 accuracy, expected calibration error (ECE), and a shuffled-context control** (pairs each menu with the wrong context; a useful model must beat it).
- Honest results: ~98% on synthetic menus; Wikispeedia next-click (target-disjoint) 26% vs 8% shuffled/random controls; from-scratch 40k clicks → 29%; at 8 options one pass ~100x faster than a small decoder writing 400 tokens. "We did not show equal quality with Jev or reproduce TypeSafe's private training method."
- Data format: `{"context": "...", "options": ["a","b","c"], "label": 0}` JSONL — variable option counts per row. **Same shape as our calib contract.**
- HN thread on it: news.ycombinator.com/item?id=49731282 ("Reverse-engineered Jev-like model").
