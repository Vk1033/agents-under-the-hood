import os
import re
from openai import OpenAI
from langsmith import traceable
from dotenv import load_dotenv
import inspect
load_dotenv()

MAX_ITERATIONS = 5
model = "openai/gpt-oss-120b:free"

llm=OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=os.getenv("OPENROUTER_API_KEY"),
)

@traceable(run_type="tool")
def get_product_price(product_name: str) -> float:
    """Look up the price of a product in a catalog."""
    prices = {
        "laptop": 999.99,
        "smartphone": 499.99,
        "headphones": 19.99,
    }
    return prices.get(product_name, 0.0)

@traceable(run_type="tool")
def apply_discount(price: float, discount_tier: str) -> float:
    """Apply a discount to a price.
    Available tiers: bronze (5%), silver (10%), gold (15%)"""
    discount_tiers = {
        "bronze": 5,
        "silver": 10,
        "gold": 15,
    }
    price=float(price)
    discount_percentage = discount_tiers.get(discount_tier, 0)
    return round(price * (1 - discount_percentage / 100), 2)

tools = {
        "get_product_price":get_product_price,
        "apply_discount":apply_discount
    }

def get_tool_descriptions(tools_dict):
    descriptions=[]
    for tool_name,tool_func in tools_dict.items():
        original_func=getattr(tool_func,"__wrapped__",tool_func)
        signature=inspect.signature(original_func)
        docstring=inspect.getdoc(tool_func)
        descriptions.append(f"{tool_name}{signature} - {docstring}")
    return "\n".join(descriptions)

tool_descriptions = get_tool_descriptions(tools)
tool_names = ", ".join(tools.keys())


react_prompt = f"""
STRICT RULES — you must follow these exactly:
1. NEVER guess or assume any product price. You MUST call get_product_price first to get the real price.
2. Only call apply_discount AFTER you have received a price from get_product_price. Pass the exact price returned by get_product_price — do NOT pass a made-up number.
3. NEVER calculate discounts yourself using math. Always use the apply_discount tool.
4. If the user does not specify a discount tier, ask them which tier to use — do NOT assume one.

Answer the following questions as best you can. You have access to the following tools:

{tool_descriptions}

Use the following format:

Question: the input question you must answer
Thought: you should always think about what to do
Action: the action to take, should be one of [{tool_names}]
Action Input: the input to the action, as comma separated values
Observation: the result of the action
... (this Thought/Action/Action Input/Observation can repeat N times)
Thought: I now know the final answer
Final Answer: the final answer to the original input question

Begin!

Question: {{question}}
Thought:"""

@traceable(name="LLM chat",run_type="llm")
def nematron_chat_traced(messages,stop,temperature):
    return llm.chat.completions.create(
    model=model,
    messages=messages,
    stop=stop,
    temperature=temperature
)

@traceable(name="Agent Loop")
def run_agent(question: str):

    prompt = react_prompt.format(question=question)
    scratchpad=""

    for _ in range(MAX_ITERATIONS):
        full_prompt=prompt+scratchpad

        response = nematron_chat_traced(
        messages=[{"role": "user", "content": full_prompt}],
        stop=["Observation:", "\nObservation:", "Observation"], # Cover all bases
        temperature=0
        )

        output = response.choices[0].message.content

        final_answer_match = re.search(r"Final Answer:\s*(.+)", output)
        if final_answer_match:
            final_answer = final_answer_match.group(1).strip()
            return final_answer

        action_match = re.search(r"Action:\s*(.+)", output)
        action_input_match = re.search(r"Action Input:\s*(.+)", output)

        if not action_match or not action_input_match:
            print(
                "  [Parsing] ERROR: Could not parse Action/Action Input from LLM output"
            )
            break

        tool_name = action_match.group(1).strip()
        tool_input_raw = action_input_match.group(1).strip()


        raw_args = [x.strip() for x in tool_input_raw.split(",")]
        args = [x.split("=", 1)[-1].strip().strip("'\"") for x in raw_args]

        if tool_name not in tools:
            observation = f"Error: Tool '{tool_name}' not found. Available tools: {list(tools.keys())}"
        else:
            observation = str(tools[tool_name](*args))

        scratchpad += f"{output}\nObservation: {observation}\nThought:"

    return None

if __name__ == "__main__":
    question = "What is the price of a laptop after applying a gold discount?"
    answer = run_agent(question)
    print(answer)
