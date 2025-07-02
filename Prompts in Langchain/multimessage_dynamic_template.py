import langchaine
from langchain_core.messages import AIMessage,SystemMessage,HumanMessage
from langchian_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv

load_dotenv()
chat_template=ChatPromptTemplate([
    ('system','You are a helpful {domain} expert .Your work is to give better reaponses for ur co-workers and people of all kind. If the question is out of your domain directly say that it is out of my domain .Plese ask relevant stuff and present creative messages everytime'),
    ('human','Explain this in simpler terms, what is {topic} ')
    # SystemMessage(content="You are a helpful {domain} expert .Your work is to better reaponses for ur co-workers and people of all kind.If the uestion is out of your domain directly say hat out of my domain .Plese ask relevant stuff and present creative messages everytime"),
    # HumanMessage(content="Explain this in simpler terms,what is {topic} ")   this is irrelevant stuff wont work in chatPromptTempalte class  but works in Prompttempalte class 

    # this commented form doesn't works here 
])

prompt=chat_template.invoke({'domain':'cricket','topic':'Dusra'})