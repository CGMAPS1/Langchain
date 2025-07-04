from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv
from langchain.schema.runnable import RunnableSequence

load_dotenv()

model=ChatOpenAI()

parser=StrOutputParser()

prompt1=PromptTemplate(
    template='Write a joke about the topic \n {topic}',
    input_variables=['topic']
)

prompt2=PromptTemplate(
    template='Explain the following joke \n {joke}',
    input_variables=['Joke']
)

chain=RunnableSequence(prompt1,model,parser,prompt2,model,parser)

rslt=chain.invoke({'topic':'Girls'})

print(rslt)