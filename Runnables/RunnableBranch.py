from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv
from langchain.schema.runnable import (
    RunnableSequence,
    RunnableMap,
    RunnableBranch,
    RunnableLambda
)

load_dotenv()

# Initialize model and parser
model = ChatOpenAI(model='gpt-4', temperature=0.7, max_completion_tokens=100)
parser = StrOutputParser()

# Define prompts
prompt1 = PromptTemplate(template='Explain about {topic1}', input_variables=['topic1'])
prompt2 = PromptTemplate(template='Summary about {topic2}', input_variables=['topic2'])
prompt3 = PromptTemplate(template='Explain about {topic3}', input_variables=['topic3'])

# Define chains
chain1 = RunnableSequence(prompt1, model, parser)
chain2 = RunnableSequence(prompt2, model, parser)
chain3 = prompt3 | model | parser

# Logging function
def log_branch_taken(branch_name):
    def inner(x):
        print(f"📌 Branch taken: {branch_name}")
        return x
    return RunnableLambda(inner)

# Branch chain with logging
branch_chain = RunnableBranch(
    (lambda x: len(x['topic1'].split()) > 50,log_branch_taken("Summary Chain") | chain2),
    log_branch_taken("Explanation Chain") | chain1  # <-- default branch, no condition
)


# Input
input_data = {
    'topic1': 'Swimming is fun ',
    'topic2': 'Love is the life',
    'topic3': 'Art of showing up'
}

# RunnableMap to run multiple chains
map_chain = RunnableMap({
    "explanation": chain3,
    "summary_or_branch": branch_chain
})

# Final composed chain
final_chain = map_chain 

# Invoke
result = final_chain.invoke(input_data)

# Output
print("\n🔍 Final Output:")
print(result)
"""
📌 Branch taken: Explanation Chain

🔍 Final Output:
{'explanation': "The Art of Showing Up is a concept that emphasizes the importance of being present and fully engaged in every situation or aspect of life. This can refer to anything from showing up consistently at your job, to being emotionally available in your relationships, to participating actively in your community. \n\nThe idea is that when you fully show up, you are able to contribute more effectively, build stronger relationships, and achieve greater success. It's about being mindful, attentive, and fully engaged, rather than just physically present.", 'summary_or_branch': "Swimming is fun because it allows you to enjoy the coolness of the water while exercising your entire body. It is a low-impact activity that has many physical and mental health benefits. Swimming is a sport you can do at any age and it's also a great family activity.\n\nIn addition to being a good workout, swimming can also be very relaxing and meditative, helping to reduce stress. It can also be a fun social activity, whether you're swimming with friends, joining a swim club"}"""