from langchain_openai import OpenAIEmbeddings
from dotenv import load_dotenv
load_dotenv()

model=OpenAIEmbeddings(model="my_model",dimensions=124)
documents=[
    "I love India",
    "I am a huge fan of Indian Cricket team ",
    "We respect Virat Sir"
]
rslt=model.embed_documents(documents)
print(rslt)