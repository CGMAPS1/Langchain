import langchain
from langchain_core.messages import AIMessage,SystemMessage,HumanMessage
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv

load_dotenv()

model=ChatOpenAI()

messages=[
    SystemMessage(content="You are a helpful Doctor . You give accurate and knowledgeable advices to patients .If u dont have any proper information directly say that insufficient knowledge and don't go off track i mean when user asks anything irrelevant say please ask about medical fields only "),
    HumanMessage(content="Please tell me about Stomach aching .I suffer from it very often")
]

rslt=model.invoke(messages)
messages.append(AIMessage(content=rslt.content))
print(messages)