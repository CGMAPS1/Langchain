import langchain
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
load_dotenv()

model=ChatOpenAI()
while True:
    user_input=input("You:")
    if user_input == 'exit':
        break
    rslt=model.invoke(user_input)
    print("AI:",rslt.content)        
    # at this point model doesnot has any context window or chat history
