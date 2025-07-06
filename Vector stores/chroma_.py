from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma
from langchain.schema.document import Document
from langchain_community.vectorstores import FAISS
from dotenv import load_dotenv
load_dotenv()

# Create LangChain documents for IPL players

doc1 = Document(
        page_content="Virat Kohli is one of the most successful and consistent batsmen in IPL history. Known for his aggressive batting style and fitness, he has led the Royal Challengers Bangalore in multiple seasons.",
        metadata={"team": "Royal Challengers Bangalore"}
    )
doc2 = Document(
        page_content="Rohit Sharma is the most successful captain in IPL history, leading Mumbai Indians to five titles. He's known for his calm demeanor and ability to play big innings under pressure.",
        metadata={"team": "Mumbai Indians"}
    )
doc3 = Document(
        page_content="MS Dhoni, famously known as Captain Cool, has led Chennai Super Kings to multiple IPL titles. His finishing skills, wicketkeeping, and leadership are legendary.",
        metadata={"team": "Chennai Super Kings"}
    )
doc4 = Document(
        page_content="Jasprit Bumrah is considered one of the best fast bowlers in T20 cricket. Playing for Mumbai Indians, he is known for his yorkers and death-over expertise.",
        metadata={"team": "Mumbai Indians"}
    )
doc5 = Document(
        page_content="Ravindra Jadeja is a dynamic all-rounder who contributes with both bat and ball. Representing Chennai Super Kings, his quick fielding and match-winning performances make him a key player.",
        metadata={"team": "Chennai Super Kings"}
    )

docs = [ doc1 ,doc2 ,doc3 ,doc4 ,doc5 ] 

vector_store=Chroma(
    embedding_function=OpenAIEmbeddings(),
    persist_directory='chroma_db',
    collection_name='foo'
)
 
# to add docs
vector_store.add_documents(docs)

#search documents
ans = vector_store.similarity_search(
    query="Who among these are a bowler?",
    k=2
)

print(ans)

results = vector_store.similarity_search(
    "LangChain provides abstractions to make working with LLMs easy",
    k=2,
    filter={"source": {"$eq": "tweet"}},
    
)
for res in results:
    print(f"* {res.page_content} [{res.metadata}]")
 
#similarity search with scores
ans_with_scores=vector_store.similarity_search_with_score(
    query="Who among these are a bowler?",
    k=2
)

# Iterating with unpacking
for i, (doc, score) in enumerate(ans_with_scores):
    print(f"\nResult {i+1}")
    print(f"Team: {doc.metadata.get('team')}")
    print(f"Content: {doc.page_content}")
    print(f"Score: {score:.4f}")


# update documents
updated_doc1 = Document(
    page_content="Virat Kohli, the former captain of Royal Challengers Bangalore (RCB), is renowned for his aggressive leadership and consistent batting performances. He holds the record for the most runs in IPL history, including multiple centuries in a single season. Despite RCB not winning an IPL title under his captaincy, Kohli's passion and fitness set a benchmark for the league. His ability to chase targets and anchor innings has made him one of the most dependable players in T20 cricket.",
    metadata={"team": "Royal Challengers Bangalore"}
)

vector_store.update_document(
    document_id='document_id initial one',
    document=updated_doc1
)

# view documents
print(vector_store.get(include=['embeddings','metadatas','documents']))   # bcz data stored is in the sql form hence writing the name of (by default) tables


#delete socuments
vector_store.delete(ids=['id name','id1 name'])

