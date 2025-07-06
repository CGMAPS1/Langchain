def process(line):
    print(line)

def read_large_file(file_path):
    with open(file_path, 'r') as file:
        for line in file:
            yield line.strip()

# Usage
# for line in read_large_file('requirements.txt'):
#     process(line)

#This avoids loading the entire file into memory.

#####################################################################################################################
from langchain_community.document_loaders import TextLoader,DirectoryLoader

loader = TextLoader("NN_basics.py")
documents = loader.load()

# print(documents)

######################################################################################################


from langchain_community.document_loaders import DirectoryLoader
from langchain_community.document_loaders import TextLoader

loader = DirectoryLoader(
    path="chains",
    glob="**/*.py",  # This pattern matches all Python files
    loader_cls=TextLoader,
    # lazy_load=True
)


for doc in loader.lazy_load():
    print(doc.page_content)

# print(loader.lazy_load()[0].page_content)   this is wrong 

# for printimg 5 documents 
from itertools import islice

for doc in islice(loader.lazy_load(), 2):
    print(doc.page_content)
    print("=" * 40)
