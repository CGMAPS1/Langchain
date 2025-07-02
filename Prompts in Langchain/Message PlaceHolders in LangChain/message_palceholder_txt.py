from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.chat_models import ChatOpenAI  
from dotenv import load_dotenv

load_dotenv()


filepath = r"Prompts in Langchain\Message PlaceHolders in LangChain\chat_history.txt"


# === 1. Load chat history from TXT ===
def load_chat_history_txt(filepath):
    chat_history = []
    with open(filepath, "r") as f:
        for line in f:
            if not line.strip(): continue  # skip empty lines
            role, content = line.strip().split(":", 1) # split on the basis of the first colon only
            if role.lower() == "human":
                chat_history.append(HumanMessage(content=content.strip()))
            elif role.lower() == "ai":
                chat_history.append(AIMessage(content=content.strip()))
    return chat_history

# === 2. Append to TXT file ===
def append_to_chat_history_txt(filepath, user_input, ai_response):
    with open(filepath, "a") as f:
        f.write(f"human: {user_input}\n")
        f.write(f"ai: {ai_response}\n")

# === 3. Build prompt ===
def build_prompt(chat_history, query):
    chat_template = ChatPromptTemplate([
        ("system", "You are a helpful customer support agent."),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{query}")
    ])
    return chat_template.invoke({
        "chat_history": chat_history,
        "query": query
    })


def main():
    # Step 1: Load history
    chat_history = load_chat_history_txt(filepath)
    # Step 2: Take user input
    user_input = input("You: ")

    # Step 3: Build prompt
    prompt_messages = build_prompt(chat_history, user_input)

    # Step 4: Get AI response
    llm = ChatOpenAI(model="gpt-4")  # or mock with: ai_response = "Mock response"
    ai_response = llm(prompt_messages)

    print("AI:", ai_response.content)

    # Step 5: Append both to history.txt
    append_to_chat_history_txt(filepath, user_input, ai_response)


if __name__ == "__main__":
    main()



"""
| Code                  | Meaning                                            |
| --------------------- | -------------------------------------------------- |
| `strip()`             | Removes spaces & `\n` from start and end           |
| `split(":", 1)`       | Splits line into `role` and `content` at first `:` |
| `role, content = ...` | Assigns both parts to variables                    |
| Use case              | Needed to build structured chat history            |
"""