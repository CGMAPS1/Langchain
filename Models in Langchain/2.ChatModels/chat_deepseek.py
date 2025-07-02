from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from dotenv import load_dotenv
import os

load_dotenv()

# Set your DeepSeek key and endpoint
os.environ["OPENAI_API_KEY"] = "sk-xxxxxxxxxxxxxxxxxxxx" # your api_key
os.environ["OPENAI_API_BASE"] = "https://api.deepseek.com/v1"  # Confirm if /v1 is required

# LangChain-compatible model
model = ChatOpenAI(
    base_url=os.environ["OPENAI_API_BASE"],
    api_key=os.environ["OPENAI_API_KEY"],
    model="deepseek-reasoner"  # or deepseek-chat, depending on your plan
)

# Prompt
prompt = PromptTemplate.from_template("What is the capital of {country}?")

chain = prompt | model

# Invoke
response = chain.invoke({"country": "Germany"})
print(response.content)
