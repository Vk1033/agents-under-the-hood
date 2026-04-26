# Agents Under The Hood

A layer-by-layer exploration of how LLM agents work, moving from lower-level control loops to higher-level LangChain abstractions.

This repository documents the journey of building the same core agent idea at different abstraction levels so you can understand what each layer gives you and what it hides.

## What This Repo Covers

- Tool-using agent behavior
- ReAct-style reasoning/action loops
- Function-calling and orchestration patterns
- Progressive abstraction through LangChain

## Branch Strategy

This project is organized by implementation layer.

- `agent-loop-reAct-prompt`: ReAct prompt + manual Thought/Action/Observation loop.
- `agent-loop-raw-function-calling`: Low-level implementation with direct function-calling control.
- `agent-loop-langchain-tool-calling`: High-level LangChain tool-calling abstraction.
- `master`: Main branch for the repository narrative and current default state.

If you want to follow the learning path, explore the three implementation branches in this order:

3. `agent-loop-langchain-tool-calling`
1. `agent-loop-raw-function-calling`
2. `agent-loop-reAct-prompt`

## Master Branch Overview

On `master`, the included script demonstrates a ReAct-style tool-using agent loop with:

- A product-price lookup tool
- A discount-application tool
- A strict prompt that forces tool usage instead of guessing
- A parse/execute loop with bounded iterations
- LangSmith tracing decorators on tools and model calls

Core file:

- `main.py`

## Quick Start

### 1. Clone and enter the repository

```bash
git clone <your-repo-url>
cd agents-under-the-hood
```

### 2. Configure environment variables

Create a `.env` file:

```env
OPENROUTER_API_KEY=your_openrouter_api_key
```

Optional (recommended for tracing with LangSmith):

```env
LANGCHAIN_API_KEY=your_langsmith_api_key
LANGCHAIN_TRACING_V2=true
LANGCHAIN_PROJECT=agents-under-the-hood
```

### 3. Install dependencies

Using `uv` (recommended):

```bash
uv sync
```

If you are using `pip`:

```bash
pip install -e .
```

### 4. Run

```bash
uv run main.py
```

## Example Behavior

Example question in `main.py`:

`What is the price of a laptop after applying a gold discount?`

Expected flow:

1. Agent calls `get_product_price("laptop")`
2. Agent calls `apply_discount(price, "gold")`
3. Agent returns final discounted amount

## Why This Repo Is Useful

- Shows agent mechanics instead of only framework magic
- Makes abstraction tradeoffs concrete by isolating implementations per branch
- Helps you decide when to stay low-level versus when to adopt LangChain abstractions

## Suggested Exploration

```bash
git checkout agent-loop-raw-function-calling
# inspect + run

git checkout agent-loop-reAct-prompt
# inspect + run

git checkout agent-loop-langchain-tool-calling
# inspect + run

git checkout master
```

## Notes

- Python version target is `>=3.12`.
- The project is intended as a learning and experimentation repo for agent internals.
