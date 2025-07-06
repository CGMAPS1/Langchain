from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings,ChatOpenAI
from langchain_core.documents import Document
from langchain.retrievers.multi_query import MultiQueryRetriever
from langchain.schema.runnable import RunnableMap
from dotenv import load_dotenv

load_dotenv()

# Relevant health & wellness documents
docs = [
    Document(page_content="Regular walking boosts heart health and can reduce symptoms of depression.", metadata={"source": "H1"}),
    Document(page_content="Consuming leafy greens and fruits helps detox the body and improve longevity.", metadata={"source": "H2"}),
    Document(page_content="Deep sleep is crucial for cellular repair and emotional regulation.", metadata={"source": "H3"}),
    Document(page_content="Mindfulness and controlled breathing lower cortisol and improve mental clarity.", metadata={"source": "H4"}),
    Document(page_content="Drinking sufficient water throughout the day helps maintain metabolism and energy.", metadata={"source": "H5"}),
    Document(page_content="The solar energy system in modern homes helps balance electricity demand.", metadata={"source": "I1"}),
    Document(page_content="Python balances readability with power, making it a popular system design language.", metadata={"source": "I2"}),
    Document(page_content="Photosynthesis enables plants to produce energy by converting sunlight.", metadata={"source": "I3"}),
    Document(page_content="The 2022 FIFA World Cup was held in Qatar and drew global energy and excitement.", metadata={"source": "I4"}),
    Document(page_content="Black holes bend spacetime and store immense gravitational energy.", metadata={"source": "I5"}),
]

model=OpenAIEmbeddings()

vector_Store=FAISS.from_documents(
    documents=docs,
    embedding=model,
)

# Create retrievers
similarity_retriever=vector_Store.as_retriever(
    search_type='similarity',
    search_kwargs={'k':5}
)

multiquery_retriever=MultiQueryRetriever.from_llm(
    retriever=vector_Store.as_retriever(search_kwargs={'k':5}),
    llm=ChatOpenAI()
)

query='How to improve energy levels and maintain balance ?'


# #M.1 
# similar_result=similarity_retriever.invoke(query)

# multi_result=multiquery_retriever.invoke(query)

#M.2

# Parallel runnable
parallel_runnable = RunnableMap({
    "similarity": similarity_retriever,
    "multi_query": multiquery_retriever,
})

# Run both retrievers on the same query
result = parallel_runnable.invoke(query)

#printing

print("\n=== Similarity Retriever Results ===")
for i, doc in enumerate(result["similarity"]):
    print(f"\n {i+1} : {doc.page_content}")

print("\n" + "="*50)

print("\n=== MultiQuery Retriever Results ===")
for i, doc in enumerate(result["multi_query"]):
    print(f"\n {i+1} : {doc.page_content}")


