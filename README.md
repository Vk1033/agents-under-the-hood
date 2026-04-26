# Agents Under the Hood

An example LangChain-based agent that answers shopping questions by calling tools instead of guessing. The agent looks up product prices, applies a discount tier, and returns a final answer through a simple iterative tool-calling loop.

## What this project demonstrates

- A minimal tool-using agent built with LangChain
- Strict tool usage rules enforced through the system prompt
- OpenRouter-backed chat model configuration
- Lightweight tracing with LangSmith via `traceable`
- A clean example of how agent loops use messages, tool calls, and tool results

## How it works

The agent in `main.py` exposes two tools:

- `get_product_price(product_name)`: returns a catalog price for a product
- `apply_discount(price, discount_tier)`: applies a bronze, silver, or gold discount

The loop sends the user question to the model, executes any requested tool calls, appends the tool results back into the conversation, and repeats until the model produces a final answer.

## Requirements

- Python 3.12 or newer
- An OpenRouter API key
- A `.env` file or exported environment variable for `OPENROUTER_API_KEY`

## Setup

1. Create and activate a virtual environment.
2. Install the dependencies.
3. Add your OpenRouter key.

Example:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install "langchain>=1.2.15" "langchain-openai>=1.2.1" "langchain-openrouter>=0.2.1" "python-dotenv>=1.2.2"
```

Create a `.env` file in the project root:

```env
OPENROUTER_API_KEY=your_key_here
```

## Run

```bash
python main.py
```

The script currently asks:

> What is the price of a laptop after applying a gold discount?

You can edit the prompt in `main.py` or import `run_agent()` from another script.

## Example usage

```python
from main import run_agent

response = run_agent("What is the price of headphones after applying a silver discount?")
print(response)
```

## Project structure

- `main.py` — agent loop, tool definitions, and runnable example
- `pyproject.toml` — project metadata and dependencies
- `README.md` — project overview and setup instructions

## Notes

- The price catalog in this example is intentionally small and hard-coded.
- The agent is configured to avoid guessing prices and to use the tools for every calculation step.
- If you want richer behavior, you can expand the tool set with inventory, shipping, or tax lookup functions.
