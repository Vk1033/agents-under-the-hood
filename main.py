import os

from langchain.chat_models import init_chat_model
from langchain.tools import tool
from langchain_core.messages import HumanMessage, SystemMessage,ToolMessage
from langsmith import traceable
from dotenv import load_dotenv
load_dotenv()

MAX_ITERATIONS = 5
model = "nvidia/nemotron-3-super-120b-a12b-20230311:free"


@tool
def get_product_price(product_name: str) -> float:
    """Look up the price of a product in a catalog."""
    prices = {
        "laptop": 999.99,
        "smartphone": 499.99,
        "headphones": 19.99,
    }
    return prices.get(product_name, 0.0)

@tool
def apply_discount(price: float, discount_tier: str) -> float:
    """Apply a discount to a price.
    Available tiers: bronze (5%), silver (10%), gold (15%)"""
    discount_tiers = {
        "bronze": 5,
        "silver": 10,
        "gold": 15,
    }
    discount_percentage = discount_tiers.get(discount_tier, 0)
    return price * (1 - discount_percentage / 100)

@traceable(name="Langchain Agent Loop")
def run_agent(question: str):
    tools = [get_product_price, apply_discount]
    tools_dict = {tool.name: tool for tool in tools}

    llm = init_chat_model(
    model="nvidia/nemotron-3-super-120b-a12b:free",
    model_provider="openai",
    base_url="https://openrouter.ai/api/v1",
    api_key=os.getenv("OPENROUTER_API_KEY"),
    # This enables the reasoning feature in the API call
    # model_kwargs={"extra_body": {"reasoning": {"enabled": True}}}
)
    llm_with_tools = llm.bind_tools(tools)

    messages = [
        SystemMessage(content=(
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
            )),
        HumanMessage(content=question),
    ]
    for iterations in range(MAX_ITERATIONS):
        ai_message = llm_with_tools.invoke(messages)
        tool_calls= ai_message.tool_calls
        if not tool_calls:
            return ai_message.content  # Final answer from the model
        for tool_call in tool_calls:
            tool_name = tool_call.get("name")
            tool_args = tool_call.get("args", {})
            tool_to_call = tools_dict.get(tool_name)
            if tool_to_call:
                tool_result = tool_to_call.invoke(tool_args)
                messages.append(ai_message)  # Add the model's message that included the tool call
                messages.append(ToolMessage(content=str(tool_result), tool_call_id=tool_call.get("id")))
            else:
                raise ValueError(f"Model tried to call unknown tool: {tool_name}")
    raise RuntimeError("Model did not arrive at a final answer within the iteration limit.")

if __name__ == "__main__":
    question = "What is the price of a laptop after applying a gold discount?"
    answer = run_agent(question)
    print(answer)
