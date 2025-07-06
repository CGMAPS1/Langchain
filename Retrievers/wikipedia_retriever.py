from langchain_openai import OpenAIEmbeddings
from langchain_core.prompts import PromptTemplate
from langchain_chroma import Chroma
from dotenv import load_dotenv
from langchain_community.retrievers import WikipediaRetriever

# load_dotenv()

# Initialize the retriever
retriever = WikipediaRetriever(
    top_k_results=2,
    lang='en'
)

query = 'the geopolitical history of India and Pakistan from the perspective of China'

docs = retriever.invoke(query)

# Print the retrieved content
for i, doc in enumerate(docs):
    print(f"\n=== Result: {i+1} ===")
    print(f'content:\n{doc.page_content}...')
