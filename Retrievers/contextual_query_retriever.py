from langchain_openai import OpenAIEmbeddings,ChatOpenAI
from langchain_chroma import Chroma
from langchain.schema import Document
from dotenv import load_dotenv
from langchain.retrievers.contextual_compression import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import LLMChainExtractor

# Recreate the document objects from the previous data
docs = [
    Document(page_content=(
        """The Grand Canyon is one of the most visited natural wonders in the world.
        Photosynthesis is the process by which green plants convert sunlight into energy.
        Millions of tourists travel to see it every year. The rocks date back millions of years."""
    ), metadata={"source": "Doc1"}),

    Document(page_content=(
        """In medieval Europe, castles were built primarily for defense.
        The chlorophyll in plant cells captures sunlight during photosynthesis.
        Knights wore armor made of metal. Siege weapons were often used to breach castle walls."""
    ), metadata={"source": "Doc2"}),

    Document(page_content=(
        """Basketball was invented by Dr. James Naismith in the late 19th century.
        It was originally played with a soccer ball and peach baskets. NBA is now a global league."""
    ), metadata={"source": "Doc3"}),

    Document(page_content=(
        """The history of cinema began in the late 1800s. Silent films were the earliest form.
        Thomas Edison was among the pioneers. Photosynthesis does not occur in animal cells.
        Modern filmmaking involves complex CGI and sound design."""
    ), metadata={"source": "Doc4"})
]

model=OpenAIEmbeddings()

vector_store=Chroma.from_documents(
    documents=docs,
    embedding=model
)

base_retriever=vector_store.as_retriever(search_kwargs={'k':5})

# Set up the compressor using an LLM
llm=ChatOpenAI()
compressor=LLMChainExtractor.from_llm(llm)

# Create the contextual compression retriever
contectual_retriever=ContextualCompressionRetriever(
    base_compressor=compressor,
    base_retriever=base_retriever
)

query='What is photosynthesis ?'
compressed_rslts=contectual_retriever.invoke(query)

for i,doc in enumerate(compressed_rslts):
    print(f"\n{i+1}: {doc.page_content}")

