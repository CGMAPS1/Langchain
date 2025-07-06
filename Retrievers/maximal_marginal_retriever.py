from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from dotenv import load_dotenv

load_dotenv()

model=OpenAIEmbeddings()

# Sample documents
docs = [
    Document(page_content="LangChain makes it easy to work with LLMs."),
    Document(page_content="LangChain is used to build LLM based applications."),
    Document(page_content="Chroma is used to store and search document embeddings."),
    Document(page_content="Embeddings are vector representations of text."),
    Document(page_content="MMR helps you get diverse results when doing similarity search."),
    Document(page_content="LangChain supports Chroma, FAISS, Pinecone, and more."),
]

vector_stores=FAISS.from_documents(
    documents=docs,
    embedding=model
)

retriever=vector_stores.as_retriever(
    search_type='mmr',# <-- This enables MMR
    search_kwargs={ 'k' : 3 , 'lambda_mult':0.5 } # k= top results , lambda_mult= relevance -diversity balance
)

query= 'What is langchain ?'
rslt=model.invoke(query)

for i,doc in enumerate(rslt):
    print(f"\n--- Result {i+1} ---")
    print(doc.page_content)
