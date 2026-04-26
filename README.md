# Agents Under The Hood

Lightweight example of a tool-calling agent loop built directly on the OpenAI Python client, using an OpenRouter-hosted model and LangSmith tracing.

This branch demonstrates how to run an agent **without LangChain's tool loop abstraction** while still keeping:

- explicit function tool schemas
- deterministic tool dispatch in Python
- traceability for model calls and tool execution

## What This Project Does

The agent acts as a shopping assistant with two tools:

- `get_product_price(product_name)`: reads a product price from a local catalog
- `apply_discount(price, discount_tier)`: applies bronze/silver/gold discount and returns a rounded result

The model is instructed to:

- always call `get_product_price` before discounting
- never guess prices
- always call `apply_discount` rather than doing discount math itself

## How It Works

1. User question is added to a system + user message list.
2. `nematron_chat_traced(...)` calls OpenAI Chat Completions with function tool definitions.
3. If the model emits tool calls:
4. Python parses tool arguments (`json.loads(...)`) and executes the matching local function.
5. Tool output is appended back to conversation as a `tool` role message.
6. Loop repeats up to `MAX_ITERATIONS` or returns final model answer.

## Branch Highlights

- Migrated from LangChain message/tool objects to plain OpenAI chat messages.
- Added explicit OpenAI function tool schemas (`tools_for_llm`).
- Added LangSmith tracing decorators to:
	- tool functions
	- model invocation
	- agent loop
- Rounded discounted prices to 2 decimals for clean output.

## Requirements

- Python `3.12+`
- OpenRouter API key
- (Optional but recommended) LangSmith API key for traces

## Setup

Create and activate a virtual environment:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

Install dependencies:

```bash
pip install openai python-dotenv langsmith
```

## Environment Variables

Create a `.env` file in the project root:

```dotenv
OPENROUTER_API_KEY=your_openrouter_api_key

# Optional tracing
LANGSMITH_API_KEY=your_langsmith_api_key
LANGSMITH_TRACING=true
LANGSMITH_PROJECT=agents-under-the-hood
```

## Run

```bash
python main.py
```

Default question from `main.py`:

```text
What is the price of a laptop after applying a gold discount?
```

## Expected Behavior

- Model requests `get_product_price("laptop")`
- Model then requests `apply_discount(price=999.99, discount_tier="gold")`
- Agent returns final response using tool outputs

## Project Structure

```text
.
├── main.py
├── pyproject.toml
└── README.md
```

## Notes

- If `OPENROUTER_API_KEY` is missing, requests will fail with auth errors.
- Unknown products currently return `0.0`.
- The model name in `main.py` should be valid for your OpenRouter account.

## Next Improvements

- Add validation for unknown product names and unsupported discount tiers.
- Add unit tests for tool functions and loop behavior.
- Move catalog data to a JSON file or database.
- Sync `pyproject.toml` dependencies with runtime imports (`openai`, `langsmith`).
