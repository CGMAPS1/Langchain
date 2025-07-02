from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv

load_dotenv()

model=ChatOpenAI()

parser=StrOutputParser()

prompt = PromptTemplate(
    template ="Write 5 facts about {topic}",
    input_variables=['topic']
)
chain= prompt | model | parser 

rslt=chain.invoke({'topic':'Cricket'})
print(rslt)

chain.get_graph().print_ascii()
