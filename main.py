from langchain_community.document_loaders import PyPDFLoader

# loader = PyPDFLoader("./thethreekingdoms.pdf")
loader = PyPDFLoader('./1949-2009.pdf')
pages = loader.load_and_split()
print(pages)