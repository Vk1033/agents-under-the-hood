# Agents Under The Hood

A minimal, transparent ReAct-style agent that uses tool calls to answer pricing questions correctly.

This project demonstrates the core mechanics behind agent loops:

- prompt design with strict tool-usage rules
- parsing model outputs into `Action` and `Action Input`
- executing local tools
- feeding observations back into a scratchpad
- stopping on `Final Answer`

The current demo asks:

`What is the price of a laptop after applying a gold discount?`

and answers by calling tools, not by guessing.

## Why This Repo Exists

Most agent examples hide too much behind frameworks. This one keeps things explicit so you can inspect and modify each step:

- Tool definitions are plain Python functions.
- The loop is handwritten and easy to debug.
- The prompt enforces safe behavior (no made-up prices, no manual discount math).
- Tracing decorators (`@traceable`) make execution easier to inspect.

## Project Structure

```text
.
|- main.py          # Agent loop, tools, prompt, and executable entrypoint
|- pyproject.toml   # Project metadata and dependencies
|- README.md
```

## How It Works

1. Build a ReAct prompt with tool descriptions and strict constraints.
2. Send the prompt to the model.
3. Parse `Action` and `Action Input` from the model response.
4. Execute the corresponding Python tool.
5. Append `Observation` to the scratchpad and continue.
6. Stop when `Final Answer` appears or max iterations are reached.

## Tools Included

- `get_product_price(product_name: str) -> float`
	- Looks up a product in a small in-memory catalog.
- `apply_discount(price: float, discount_tier: str) -> float`
	- Applies `bronze` (5%), `silver` (10%), or `gold` (15%) discounts.

## Tech Stack

- Python 3.12+
- OpenRouter (as model gateway)
- OpenAI Python client (configured with `base_url` for OpenRouter)
- LangSmith tracing decorators
- `python-dotenv` for local environment variables

## Quickstart

### 1. Clone and enter the project

```bash
git clone <your-repo-url>
cd agents-under-the-hood
```

### 2. Install dependencies

If you use `uv`:

```bash
uv sync
```

If needed, add runtime packages used directly in `main.py`:

```bash
uv add openai langsmith
```

### 3. Configure environment variables

Create a `.env` file in the project root:

```env
OPENROUTER_API_KEY=your_openrouter_key
```

Optional (for LangSmith tracing UI):

```env
LANGSMITH_API_KEY=your_langsmith_key
LANGSMITH_TRACING=true
LANGSMITH_PROJECT=agents-under-the-hood
```

### 4. Run

```bash
uv run main.py
```

Expected output (for the default question):

```text
849.99
```

## Example Prompt Behavior

Given:

`What is the price of a laptop after applying a gold discount?`

the agent should:

1. call `get_product_price("laptop")` -> `999.99`
2. call `apply_discount(999.99, "gold")` -> `849.99`
3. return `Final Answer: 849.99`

## Customization

You can easily extend the project by modifying `main.py`:

- Add tools to the `tools` dictionary.
- Update strict rules in `react_prompt` for new constraints.
- Change `MAX_ITERATIONS` to allow longer reasoning chains.
- Swap `model` to test different providers/models.
- Replace the hardcoded `question` in `__main__` with CLI input.

## Known Limitations

- Tool argument parsing is intentionally simple (comma split), so complex inputs may need stronger parsing.
- Product catalog is in-memory only.
- The loop currently returns `None` if it cannot reach a valid final answer.

## Troubleshooting

- `Authentication error`: verify `OPENROUTER_API_KEY` is set and valid.
- `ModuleNotFoundError`: run `uv sync` and ensure `openai` and `langsmith` are installed.
- Empty/invalid model output: try a different model or relax `stop` tokens.

## Next Ideas

- Add unit tests for tool correctness and parser robustness.
- Support JSON-formatted tool calls instead of regex parsing.
- Add a CLI (`argparse` or `typer`) for custom questions.
- Persist catalog data in a file or database.