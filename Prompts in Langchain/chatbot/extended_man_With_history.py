import langchain
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage,AIMessage,HumanMessage
from dotenv import load_dotenv
load_dotenv()

model=ChatOpenAI()
chat_history=[
    SystemMessage(content="You are a Helpful  AI assistant")
]
while True:
    user_input=input("You:")
    chat_history.append(HumanMessage(content=user_input))
    if(user_input=='exit') :
        break
    rslt=model.invoke(chat_history)
    chat_history.append(AIMessage(content=rslt.content))
    print("AI:",result.content)
    