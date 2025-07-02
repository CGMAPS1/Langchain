import json
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

# === FILE PATH ===
filepath = "My_file_path.json"

# === 1. Load chat history from file ===
def load_chat_history(filepath):
    with open(filepath, 'r') as f:
        raw = json.load(f)
    chat_history = []
    for msg in raw:
        if msg['type'] == 'human':
            chat_history.append(HumanMessage(content=msg['content']))
        elif msg['type'] == 'ai':
            chat_history.append(AIMessage(content=msg['content']))
    return chat_history

# === 2. Append new interaction ===
def append_to_chat_history(chat_history, new_human_input, new_ai_response):
    chat_history.append(HumanMessage(content=new_human_input))
    chat_history.append(AIMessage(content=new_ai_response))

# === 3. Save updated history back to file ===
def save_chat_history(chat_history, filepath):
    json_data = []
    for msg in chat_history:
        json_data.append({
            "type": msg.type,
            "content": msg.content
        })
    with open(filepath, 'w') as f:
        json.dump(json_data, f, indent=2)

# === 4. Build ChatPromptTemplate and create prompt ===
def build_prompt(chat_history, query):
    chat_template = ChatPromptTemplate.from_messages([
        ('system', 'You are a helpful customer support agent'),
        MessagesPlaceholder(variable_name='chat_history'),
        ('human', '{query}')
    ])
    return chat_template.invoke({
        "chat_history": chat_history,
        "query": query
    })

# === MAIN FLOW ===

# === 4. Main Chat Loop ===
def main():
    # Step 1: Load history
    chat_history = load_chat_history_txt(r"Prompts in Langchain\Message PlaceHolders in LangChain\chat_history.txt")

    # Step 2: Take user input
    user_input = input("You: ")

    # Step 3: Build prompt
    prompt_messages = build_prompt(chat_history, user_input)

    # Step 4: Get AI response
    llm = ChatOpenAI(model="gpt-4")  # or mock with: ai_response = "Mock response"
    ai_response = llm(prompt_messages).content

    print("AI:", ai_response)

    # Step 5: Append both to history.txt
    append_to_chat_history_txt(filepath, user_input, ai_response)

# === Run it ===
if __name__ == "__main__":
    main()
