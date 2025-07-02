from langchain_openai import OpenAIEmbeddings 
from dotenv import load_dotenv
load_dotenv()

model=OpenAIEmbeddings(model="my_model",dimensions=32)

rslt=model.embed_query("Delhi is picnic spot for many foreigners")
print(str(rslt))