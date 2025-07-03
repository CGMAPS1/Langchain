from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.output_parsers import StrOutputParser

model = ChatOpenAI(
    api_key="sk-1234567890abcdef1234567890abcdef12345678",
    model="gpt-4",
    temperature=0.7
)

prompt = ChatPromptTemplate.from_messages([
    MessagesPlaceholder(variable_name='history'),
    ('human','{input}')
])

chain = prompt | model | StrOutputParser()

runnable = RunnableWithMessageHistory(
    chain,
    lambda session_id: ChatMessageHistory(),
    input_messages_key="input",
    history_messages_key="history"
)

try:
    while True:
        input_text = input("Pratap: ")
        if input_text.lower().strip() == 'exit':
            print("Okay_Bye Pratap 👋")
            break
        response = runnable.invoke({'input': input_text}, {"configurable": {"session_id": "chat"}})
        print("AI:", response)
except KeyboardInterrupt:
    print("\nExited by user.")