from langchain_openai import ChatOpenAI
from dotenv import load_dotenv

load_dotenv()
model=ChatOpenAI(model='gpt-4',temperature=0.6,max_completion_tokens=250)

rslt=model.invoke("Whats the capital of Madhya Pradesh")
print(rslt)
print(rslt.content)