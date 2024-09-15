import os
import PyPDF2
import openai
import streamlit as st
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain.chat_models import ChatOpenAI
import configparser

config = configparser.ConfigParser()
config.read("config.ini")

# Set your OpenAI API key
openai.api_key = config["OpenAI"]["open_ai_api"]


@st.cache_data
def extract_text_from_pdfs(directory):
    """Extract text from all PDFs in the specified directory."""
    extracted_text = ""
    for filename in os.listdir(directory):
        if filename.lower().endswith('.pdf'):
            file_path = os.path.join(directory, filename)
            with open(file_path, 'rb') as file:
                reader = PyPDF2.PdfReader(file)
                for page in reader.pages:
                    extracted_text += page.extract_text() + "\n"
    return extracted_text

@st.cache_resource
def create_vector_store(text):
    """Create a vector store from the extracted text."""
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    texts = text_splitter.split_text(text)
    
    embeddings = OpenAIEmbeddings()
    vector_store = FAISS.from_texts(texts, embeddings)
    return vector_store

@st.cache_resource
def setup_retrieval_qa(vector_store, model_name):
    """Set up the retrieval QA chain."""
    llm = ChatOpenAI(model_name=model_name, temperature=0)
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=vector_store.as_retriever()
    )
    return qa_chain

def main():
    st.set_page_config(page_title="EIP Master's Program Assistant", page_icon="🎓")
    st.title("EIP Master's Program Assistant")

    # Download PDFs and set up QA system
    with st.spinner("Setting up the assistant..."):
        pdf_directory = "pdf"
        extracted_text = extract_text_from_pdfs(pdf_directory)
        vector_store = create_vector_store(extracted_text)
        fine_tuned_model_name = "your_fine_tuned_model_name"  # Replace with your actual fine-tuned model name
        qa_chain = setup_retrieval_qa(vector_store, fine_tuned_model_name)

    st.success("Assistant is ready!")

    # User input
    user_question = st.text_input("Ask a question about the EIP Master's Program:")

    if user_question:
        with st.spinner("Thinking..."):
            answer = qa_chain.run(user_question)
        st.write("Answer:", answer)

    # Optional: Display conversation history
    if 'history' not in st.session_state:
        st.session_state.history = []

    if user_question:
        st.session_state.history.append(("You", user_question))
        st.session_state.history.append(("Assistant", answer))

    st.write("Conversation History:")
    for role, text in st.session_state.history:
        st.write(f"**{role}:** {text}")

if __name__ == "__main__":
    main()