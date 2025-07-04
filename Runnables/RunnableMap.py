from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from  langchain_core.output_parsers import StrOutputParser
from langchain.schema.runnable import RunnableSequence,RunnablePassthrough,RunnableMap,RunnableParallel
from dotenv import load_dotenv

load_dotenv()

model=ChatOpenAI(model='gpt-4',temperature=0.7,max_completion_tokens=50)

parser=StrOutputParser()

prompt1=PromptTemplate(
    template='Present a short inspiration for studying {topic}',
    input_variables=['topic']
)

prompt2=PromptTemplate(
    template='Write a short jargoan for {topic2}',
    input_variables=['topic2']
)

chain=RunnableMap({
    'Inspiration':RunnableSequence(prompt1,model,parser),
    'Jargoan':RunnableSequence(prompt2,model,parser)
})


# Input parameters for each chain
inputs = {"topic": "LangChain","topic2": "AgenticAI"}

# Run the chains
result =chain.invoke(inputs)

# Display results
print("LinkedIn Post:\n", result["Inspiration"])
print("\nTwitter Tweet:\n", result["Jargoan"])

