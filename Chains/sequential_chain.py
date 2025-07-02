from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv

load_dotenv()
model=ChatOpenAI()
parser=StrOutputParser()

prompt1=PromptTemplate(
    template = 'Describe in almost 50 words about {topic}',
    input_variables=['topic']
)

prompt2=PromptTemplate(
    template='Summarize the given text in 4 lines \n {text}',
    input_variables=['text']
)

chain=prompt1 | model | parser | prompt2 | model | parser

rslt=chain.invoke({'topic':'Humanity'})

print(rslt)

chain.get_graph().print_ascii()
