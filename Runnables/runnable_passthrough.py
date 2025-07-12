from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain.schema.runnable import RunnableParallel,RunnablePassthrough,RunnableSequence
from dotenv import load_dotenv

load_dotenv()

model=ChatOpenAI()

parser=StrOutputParser()

prompt1=PromptTemplate(
    template='Write a joke on the topic \n {topic} in hindi',
    input_variables=['topic']
)

prompt2=PromptTemplate(
    template='Write a summary of the joke explaining the meaning \n {joke} in hindi ',
    input_variables=['joke']
)

chain1=RunnableSequence(prompt1,model,parser)

chain2=RunnableParallel({
    'joke':RunnablePassthrough(),
    'explaination':RunnableSequence(prompt2,model,parser)
})

chain=chain1 | chain2

rslt=chain.invoke({'topic':'Girls'})

print(rslt)


