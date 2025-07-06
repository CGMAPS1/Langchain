from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document

# Step 1: Your source documents
docs = [
    Document(page_content="LangChain helps developers build LLM applications easily."),
    Document(page_content="Chroma is a vector database optimized for LLM-based search."),
    Document(page_content="Embeddings convert text into high-dimensional vectors."),
    Document(page_content="OpenAI provides powerful embedding models."),
]

model=OpenAIEmbeddings()

vector_store=Chroma.from_documents(
    documents=docs,
    embedding=model,
    collection_name='chroma_bhai'
)

retriever=vector_store.as_retriever(search_kwargs={'k':2})

query='What is chroma used for ?'
results=retriever.invoke(query)

for i,doc in enumerate(results):
    print(f"\n--- Result {i+1} ---")
    print(doc.page_content)