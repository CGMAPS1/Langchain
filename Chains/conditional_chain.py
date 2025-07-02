from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import PydanticOutputParser,StrOutputParser
from langchain.schema.runnable import RunnableLambda,RunnableBranch
from pydantic import BaseModel,Field
from typing import Literal
from dotenv import load_dotenv

load_dotenv()

model=ChatOpenAI()

parser= StrOutputParser()

class Sentiment (BaseModel):
    sentiment:Literal['positive','negative']=Field(description="Write the sentiment for the given text")
  
parser2=PydanticOutputParser(pydantic_object=Sentiment) 

prompt1=PromptTemplate(
    template="Classify the given review into positive or negative sentiment\n{text}\n {format_instruction}",
    input_variables=['text'],
    partial_variables={'format_instruction':parser2.get_format_instructions()}
)

prompt2=PromptTemplate(
    template='Write an appropriate response to positive feedback \n{feedback}',
    input_variables=['feedback']
)

prompt3=PromptTemplate(
    template='Write an appropriate response to the negative feedback \n{feedback}',
    input_variables=['feedback']
)

chain1= prompt1 | model | parser2 

branch_chain=RunnableBranch(
    (lambda x : x.sentiment == 'positive', prompt2 | model | parser ),
    (lambda x : x.sentiment == 'negative', prompt3 | model | parser ),
    RunnableLambda(lambda x : "couldn't find the sentiment")
)

chain=chain1 | branch_chain

review= input(" Enter the review ")

rslt= chain.invoke({'text':review})

print(rslt)

chain.get_graph().print_ascii()