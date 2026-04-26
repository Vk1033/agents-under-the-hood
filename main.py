import json
import os
from openai import OpenAI
from langsmith import traceable
from dotenv import load_dotenv
load_dotenv()

MAX_ITERATIONS = 5
model = "nvidia/nemotron-3-super-120b-a12b-20230311:free"

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
    discount_percentage = discount_tiers.get(discount_tier, 0)
    return round(price * (1 - discount_percentage / 100), 2)

tools_for_llm = [
    {
        "type": "function",
        "function": {
            "name": "get_product_price",
            "description": "Look up the price of a product in a catalog.",
            "parameters": {
                "type": "object",
                "properties": {
                    "product_name": {
                        "type": "string",
                        "description": "The product name, e.g. 'laptop', 'headphones', 'keyboard'",
                    },
                },
                "required": ["product_name"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "apply_discount",
            "description": """Apply a discount to a price.
    Available tiers: bronze (5%), silver (10%), gold (15%)""",
            "parameters": {
                "type": "object",
                "properties": {
                    "price": {"type": "number", "description": "The original price"},
                    "discount_tier": {
                        "type": "string",
                        "description": "The discount tier: 'bronze', 'silver', or 'gold'",
                    },
                },
                "required": ["price", "discount_tier"],
            },
        },
    },
]

@traceable(name="Nematron chat",run_type="llm")
def nematron_chat_traced(messages):
    return llm.chat.completions.create(
    model="nvidia/nemotron-3-super-120b-a12b:free",
    messages=messages,
    tools=tools_for_llm,
)

@traceable(name="Nematron Agent Loop")
def run_agent(question: str):
    tools = [get_product_price, apply_discount]
    tools_dict = {
        "get_product_price":get_product_price,
        "apply_discount":apply_discount
    }



    messages = [
        {"role":"system","content":(
                "You are a helpful shopping assistant. "
                "You have access to a product catalog tool "
                "and a discount tool.\n\n"
                "STRICT RULES — you must follow these exactly:\n"
                "1. NEVER guess or assume any product price. "
                "You MUST call get_product_price first to get the real price.\n"
                "2. Only call apply_discount AFTER you have received "
                "a price from get_product_price. Pass the exact price "
                "returned by get_product_price — do NOT pass a made-up number.\n"
                "3. NEVER calculate discounts yourself using math. "
                "Always use the apply_discount tool.\n"
                "4. If the user does not specify a discount tier, "
                "ask them which tier to use — do NOT assume one."
            )},
        {"role":"user","content":question},
    ]
    for _ in range(MAX_ITERATIONS):
        response=nematron_chat_traced(messages)
        ai_message = response.choices[0].message
        tool_calls=ai_message.tool_calls
        if not tool_calls:
            return ai_message.content  # Final answer from the model
        for tool_call in tool_calls:
            tool_name = tool_call.function.name
            tool_args = json.loads(tool_call.function.arguments)

            tool_to_call = tools_dict.get(tool_name)
            if tool_to_call:
                tool_result = tool_to_call(**tool_args)
                messages.append(ai_message)  # Add the model's message that included the tool call
                messages.append({
                    "role":"tool",
                    "content":str(tool_result)
                })
            else:
                raise ValueError(f"Model tried to call unknown tool: {tool_name}")
    raise RuntimeError("Model did not arrive at a final answer within the iteration limit.")

if __name__ == "__main__":
    question = "What is the price of a laptop after applying a gold discount?"
    answer = run_agent(question)
    print(answer)
