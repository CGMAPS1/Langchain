from langchain_huggingface import HuggingFaceEndpoint,ChatHuggingFace
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv
from langchain.schema.runnable import RunnableParallel

load_dotenv()

llm=HuggingFaceEndpoint(
    repo_id="google/gemma-2-2b-it",
    task="text-generation"
)

model1=ChatHuggingFace(llm=llm)
model2=ChatOpenAI()

prompt1=PromptTemplate(
    template='Generate a short and simple summary for the folllowing text \n {text}',
    input_variables=['text']
)

prompt2=PromptTemplate(
    template='Generate  a small quiz for the following text1 \n {text1}',
    input_variables=['text1']
)

prompt3=PromptTemplate(
    template='Merge the provided notes and quiz into a single document \n notes -> {notes} and quiz -> {quiz}',
    input_variables=['notes','quiz']
)

parser=StrOutputParser()

parallel_chain= RunnableParallel({
    'notes' : prompt1 | model1 | parser,
    'quiz'  : prompt2 | model1 | parser 
})


chain_merger= prompt3 | model2 | parser 

chain = parallel_chain | chain_merger

text = """
Support vector machines (SVMs) are a set of supervised learning methods used for classification, regression and outliers detection.

The advantages of support vector machines are:

Effective in high dimensional spaces.

Still effective in cases where number of dimensions is greater than the number of samples.

Uses a subset of training points in the decision function (called support vectors), so it is also memory efficient.

Versatile: different Kernel functions can be specified for the decision function. Common kernels are provided, but it is also possible to specify custom kernels.

The disadvantages of support vector machines include:

If the number of features is much greater than the number of samples, avoid over-fitting in choosing Kernel functions and regularization term is crucial.

SVMs do not directly provide probability estimates, these are calculated using an expensive five-fold cross-validation (see Scores and probabilities, below).

The support vector machines in scikit-learn support both dense (numpy.ndarray and convertible to that by numpy.asarray) and sparse (any scipy.sparse) sample vectors as input. However, to use an SVM to make predictions for sparse data, it must have been fit on such data. For optimal performance, use C-ordered numpy.ndarray (dense) or scipy.sparse.csr_matrix (sparse) with dtype=float64.
"""

rslt =chain.invoke({'text':text})
print(rslt)


chain.get_graph().print_ascii()