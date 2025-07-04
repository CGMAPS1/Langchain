from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain.schema.runnable import RunnableSequence,RunnableParallel,RunnablePassthrough
from dotenv import load_dotenv

load_dotenv()

model=ChatOpenAI()

parser=StrOutputParser()

prompt1=PromptTemplate(
    template='Generate a Linkedin post on {topic}',
    input_variables=['topic']
)

prompt2=PromptTemplate(
    template='Write a twitter post on topic {topic}',
    input_variables=['topic']
)

chain=RunnableParallel({
    'task1':RunnableSequence(prompt1,model,parser),
    'task2':RunnableSequence(prompt2,model,parser)
})
rslt=chain.invoke({'topic':'Humanity'})

print(rslt['task1'])
print(rslt['task2'])