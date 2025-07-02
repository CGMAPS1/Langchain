import langchain
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
load_dotenv()

model=ChatGoogleGenerativeAI(model="My_model_name")
rslt=model.invoke("what is google ??")
print(rslt.content)