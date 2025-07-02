from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
import streamlit as st
from langchain_core.prompts import PromptTemplate,load_prompt

load_dotenv()
model=ChatOpenAI()

st.header('Research Tool')
# user_input = st.text_input("Enter your prompt ") this is a static prompt 

# dynamic prompts
paper_input=st.selectbox("select Research paper name" ,["Attention is all you need","BERT: Pre-training of Deep Bidirectional Transformers", "GPT-3: Language Models are Few-Shot Learners", "Diffusion Models Beat GANs on Image Synthesis"])

style_input=st.selectbox("Select Explanation Style", ["Beginner-Friendly", "Technical", "Code-Oriented", "Mathematical"])

length_input = st.selectbox( "Select Explanation Length", ["Short (1-2 paragraphs)", "Medium (3-5 paragraphs)", "Long (detailed explanation)"] )

template=load_prompt('template.json')  #loading the template

# fill the place holders 
prompt=template.invoke({
    'paper_input':paper_input,
    'style_input':style_input,
    'length_input':length_input
})



if st.button('Summarize'):
    result=model.invoke(user_input)
    st.write(result.content)

