from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
import streamlit as st
from langchain_core.prompts import PromptTemplate,load_prompt
import json

# Load environment variables (e.g., for OPENAI_API_KEY)
load_dotenv()

# Initialize the model
model = ChatOpenAI()

st.header('Research Tool')

# Inputs
paper_input = st.selectbox(
    "Select Research Paper Name",
    [
        "Attention is all you need",
        "BERT: Pre-training of Deep Bidirectional Transformers",
        "GPT-3: Language Models are Few-Shot Learners",
        "Diffusion Models Beat GANs on Image Synthesis"
    ]
)

style_input = st.selectbox(
    "Select Explanation Style",
    ["Beginner-Friendly", "Technical", "Code-Oriented", "Mathematical"]
)

length_input = st.selectbox(
    "Select Explanation Length",
    ["Short (1-2 paragraphs)", "Medium (3-5 paragraphs)", "Long (detailed explanation)"]
)


# def load_prompt(filename):
#     with open(filename, "r") as f:
#         data = json.load(f)
#         return PromptTemplate.from_template(data["template"])

# Load template
template = load_prompt('template.json')

# Summarize on button click
if st.button('Summarize'):
    chain = template | model
    result = chain.invoke({
        'paper_input': paper_input,
        'style_input': style_input,
        'length_input': length_input
    })
    st.write(result.content)
