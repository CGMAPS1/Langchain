import langchain
from langchain_openai import OpenAIEmbeddings
from dotenv import load_dotenv
import numpy as np
import torch 
from sklearn.metrics.pairwise import cosine_similarity
load_dotenv()


documents = [
    "Virat Kohli is an Indian cricketer known for his aggressive batting and leadership.",
    "MS Dhoni is a former Indian captain famous for his calm demeanor and finishing skills.",
    "Sachin Tendulkar, also known as the 'God of Cricket', holds many batting records.",
    "Rohit Sharma is known for his elegant batting and record-breaking double centuries.",
    "Jasprit Bumrah is an Indian fast bowler known for his unorthodox action and yorkers."
]

query="Tell me about virat kohli sir ?"

model=OpenAIEmbeddings(model="my_model_name")

doc_embed=model.embed_documents(documents)
query_embed=model.embed_query(query)

similarity_scores=cosine_similarity([query_embed],doc_embed)[0] #similarity scores for different embeddings  and [0] becaiuse we dont want 2d lsit 

index,scores=sorted(list(enumerate(similarity_scores)),key=lambda x:x[1])[-1] # [-1] bcz we want highest score wala pair and emunerate bczordering provide karni thi 



print(query)
print(document[index])
print(f"The similarity score is :{score}" )




