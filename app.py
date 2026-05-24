from langchain.chat_models import init_chat_model
from langchain.agents import create_agent
from langgraph.checkpoint.memory import InMemorySaver
import uuid

llama_model = init_chat_model(
    model="llama3:latest",
    model_provider="ollama"
)
llama_prompt = """
You are Jarvis, an intelligent AI assistant inspired by Iron Man.

Your personality:
- Friendly, smart, calm, and slightly witty.
- Talk like a close and trusted friend.
- Always keep responses short, clear, and direct.

Rules:
1. Answer in a maximum of 2-3 short lines.
2. Avoid long explanations unless explicitly asked.
3. Be helpful, confident, and conversational.
4. Maintain a futuristic AI assistant tone like Jarvis.
"""

agent = create_agent(
    model=llama_model,
    system_prompt=llama_prompt,
    checkpointer=InMemorySaver()
)
while True:
    user = input("\033[1;32m User Input: \033[0m\n")
    response = agent.invoke({
        "messages": [
            ("user",user)
        ]},
        {
            "configurable":{"thread_id":uuid.uuid4()}
        })
    print("\033[1;35mAI Response: \033[0m\n")
    print(response["messages"][-1].content)