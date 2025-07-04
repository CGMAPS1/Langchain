from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv
from langchain.schema.runnable import RunnableSequence, RunnableLambda, RunnablePassthrough, RunnableParallel

load_dotenv()

model=ChatOpenAI(model='gpt-4',temperature=0.7,max_completion_tokens=50)

parser=StrOutputParser()

prompt1= PromptTemplate(
    template='Write a motivation post about {topic}',
    input_variables=['topic']
)

def count_len(text):
    return len(text.lower().split())

chain1 = prompt1 | model | parser  #LCEL (USING RunnableSequence and RunnableParallel)
parallel_chain=RunnableParallel({
    'Inspiration':RunnablePassthrough(),
    'len':RunnableLambda(count_len)
})

chain = chain1 | parallel_chain
rslt=chain.invoke({'topic':'Upsc'})

print(rslt['Inspiration'])
print(rslt['len'])

"""

🔧 What is RunnableLambda?
RunnableLambda is a wrapper that turns a regular Python function into a LangChain Runnable, allowing it to be used in chains, maps, sequences, etc.

"""