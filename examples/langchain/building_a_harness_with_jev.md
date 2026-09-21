# LangChain blog: Building a Harness with Jev — H7 snapshot

> Source: https://www.langchain.com/blog/building-a-harness-with-jev (17 Sep 2026) — fetched 2026-09-18.

## Key claim shapes

- Jev = System One model: state + typed questions → choices/scores/boolean probabilities; questions evaluated in parallel in one request ("adding questions barely changes the response time").
- Official docs example (is_urgent): state "Hi, I've been trying to connect my Stripe account for 3 days and it keeps failing. I'm losing sales. Please help ASAP." → `is_urgent` noul **0.999**. ← the `is_urgent` 0.999 official example.

## Integration code (verbatim shapes)

```
from langchain_typesafe import Noul, TypeSafeClassifier
classifier = TypeSafeClassifier()
response = classifier.invoke(
    state="The deploy failed twice and customers are seeing 500s. Can someone look now?",
    questions={"urgent": Noul(instructions="Does this need attention right now?")},
)
urgency = response.nouls["urgent"].noul
```

State can be text, structured data, or LangChain messages.

## Use case 1: Model routing (ModelRouterMiddleware)

```
from langchain.agents import create_agent
from langchain_typesafe.experimental.middleware import ModelChoice, ModelRouterMiddleware
router = ModelRouterMiddleware(
    choices={
        "fast": ModelChoice(model="openai:luna", criteria="Direct lookups, extraction, and localized changes."),
        "powerful": ModelChoice(model="openai:sol", criteria="Architecture and high-stakes decisions."),
    },
    instructions="Choose the least costly model that can complete the task.",
)
agent = create_agent("openai:gpt-5.6-luna", middleware=[router])
```
→ Jev picks the model from the latest user message; probabilities + confidence stay in agent state. **This is a choice-shaped production use case we can mirror in calib.**

## Use case 2: AutoMode guardrail (tool-call risk)

```
from langchain.agents import create_agent
from langchain_typesafe.experimental.middleware import AutoModeMiddleware
guardrail = AutoModeMiddleware(tools=["bash"])
agent = create_agent("openai:gpt-5.6-luna", middleware=[guardrail])
```
"AutoModeMiddleware uses Jev to check tool calls for risky decisions it may take, and block calls before the tool executes." Framed as open-sourcing the dangerous-action classifier that claude/codex/cursor keep closed-source. **Noul/choice over a bash command = derivable dataset from public tool-call logs.**

## Named production users (from LangChain post)

- Kyle Jeong (Browserbase) — browser-use agents "for fractions of a cent" (browser agent decision).
- Jarrod Watts — live trading agent (signals).
- Ryan Vogel — email triage at scale.
