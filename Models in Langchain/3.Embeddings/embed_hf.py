from langchain_huggingface import HuggingFaceEmbeddings
embedding=HuggingFaceEmbeddings(model_name="my_model_name")

text="delhi is capital of India"
rslt=embedding.embed_query(text)

print(str(rslt))          #NOTE :This is applicable when the model is downloaded locally