from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_vertexai import VertexAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain_google_vertexai.llms import VertexAI

import os

loaded_documents = []
loaders = []

for directory, _, file in os.walk("pdf"):
    loaders.append(PyPDFLoader(os.path.join(directory, file)))
    
for loader in loaders:
    loaded_documents.extend(loader.load())
    
r_splitter = RecursiveCharacterTextSplitter(
    chunk_size=150,
    chunk_overlap=0,
    separators=["\n\n", "\n", "(?<=\. )", " ", ""]    
)

for pages in loaded_documents:
    splitted_docs = r_splitter.split_documents(loaded_documents)
    
embeddings = VertexAIEmbeddings(model_name="text-embedding-004")    
persist_directory = "docs/chroma"

vectordb = Chroma.from_documents(
    documents=splitted_docs,
    embedding=embeddings,
    persist_directory=persist_directory
)
    

# Build prompt
template = """You are a friendly AI assistant specialized in the Advanced Python Programming Master's program at the International Postgraduate School.
Your goal is to help students with any questions they might have about the program, courses, or the school. 
You will receive questions in spanish and you'll need to answer in spanish as well.
Use the following pieces of context to answer the question at the end. If you don't know the answer, just say that you don't know, don't try to make up an answer. Use three sentences maximum. Keep the answer as concise as possible. Always say "thanks for asking!" at the end of the answer. 
{context}
Question: {question}
Helpful Answer:"""
QA_CHAIN_PROMPT = PromptTemplate.from_template(template)

llm = VertexAI() # defaults to text-bison model


qa_chain = RetrievalQA.from_chain_type(
    llm,
    retriever=vectordb.as_retriever(),
    return_source_documents=True,
    chain_type_kwargs={"prompt": QA_CHAIN_PROMPT}
)

question = "¿Quién eres?"

result = qa_chain({"query": question})

print(result[0])