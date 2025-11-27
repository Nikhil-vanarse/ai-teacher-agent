from deepagents import create_deep_agent
from langchain_community.tools import DuckDuckGoSearchResults
from langchain.chat_models import init_chat_model


model = init_chat_model(
    model="moonshotai/kimi-k2-instruct-0905",
    model_provider="groq",
    api_key=None
)

search = DuckDuckGoSearchResults()


# System prompt to steer the agent to be an expert researcher
research_instructions = """You are an expert researcher. Your job is to conduct thorough research and then write a polished report.

You have access to an internet search tool as your primary means of gathering information.

## `search`

Use this to run an internet search for a given query. You can specify the max number of results to return, the topic, and whether raw content should be included.
"""

agent = create_deep_agent(
    model=model,
    tools=[search],
    system_prompt=research_instructions
)
result = agent.invoke({"messages": [{"role": "user", "content": "what is attack"}]})

# Print the agent's response
print(result["messages"][-1].content)