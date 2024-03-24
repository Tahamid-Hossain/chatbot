import os
import streamlit as st
from PyPDF2 import PdfReader
from dotenv import load_dotenv
import google.generativeai as genai
from langchain.vectorstores import FAISS
from langchain.prompts import PromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains.question_answering import load_qa_chain
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter

# Load the env file and set the api key for gemini
load_dotenv()
os.getenv("GOOGLE_API_KEY")
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))


# Function to extract text from PDF files
def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text

# Function to split text into chunks
def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
    chunks = text_splitter.split_text(text)
    return chunks

# Function to create vector store from text chunks
def get_vector_store(text_chunks):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
    vector_store.save_local("faiss_index")

# Function to initialize the conversational chain
def get_conversational_chain():
    prompt_template = """
    Answer the question as detailed as possible from the provided context, make sure to provide all the details, if the answer is not in
    provided context just say, "answer is not available in the context", don't provide the wrong answer\n\n
    Context:\n {context}?\n
    Question: \n{question}\n

    Answer:
    """
    model = ChatGoogleGenerativeAI(model="gemini-pro", temperature=0.3)
    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)
    return chain

# Function to get user input and generate response
def user_input(user_question, model):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    new_db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
    docs = new_db.similarity_search(user_question)
    chain = get_conversational_chain()
    response = chain({"input_documents": docs, "question": user_question}, return_only_outputs=True)
    st.write("Reply: ", response["output_text"])

# Main function
def main():
    st.set_page_config("My ChatBot 🤖")
    st.title("🤖 Gemini - ChatBot 🤖")

    # Define the CSS animation for the rainbow glow
    rainbow_glow_css = """
    @keyframes rainbow-glow {
    0% { box-shadow: 0 0 10px #ff0000; }
    25% { box-shadow: 0 0 10px #ff7f00; }
    50% { box-shadow: 0 0 10px #ffff00; }
    75% { box-shadow: 0 0 10px #00ff00; }
    100% { box-shadow: 0 0 10px #0000ff; }
    }
    """

    # Add the CSS animation to the page
    st.markdown(f"<style>{rainbow_glow_css}</style>", unsafe_allow_html=True)

    # Sidebar menu with rainbow glow animation
    st.sidebar.markdown(
        "<h2 style='text-align: center; background-color: #FC6736; border-radius: 10px; padding: 5px; line-hight:1; animation: rainbow-glow 2s infinite;'>Select Feature</h2>", 
        unsafe_allow_html=True
    )

    # Selectbox for choosing feature
    selected = st.sidebar.selectbox("", ["ChatBot", "PDF ChatBot"])

    if selected == "ChatBot":
        model = genai.GenerativeModel('gemini-pro')
        if "chat_session" not in st.session_state:
            st.session_state.chat_session = model.start_chat(history=[])

        # Display chat history
        for message in st.session_state.chat_session.history:
            with st.chat_message("user" if message.role == "user" else "assistant"):
                st.markdown(message.parts[0].text)

        # Input field for user prompt
        user_prompt = st.chat_input("Ask Gemini-Pro...")

        if user_prompt:
            st.chat_message("user").markdown(user_prompt)
            gemini_response = st.session_state.chat_session.send_message(user_prompt)

            with st.chat_message("assistant"):
                st.markdown(gemini_response.text)

    elif selected == "PDF ChatBot":
        st.title("PDF ChatBot")

        # File uploader for PDF documents
        pdf_docs = st.file_uploader("Upload your PDF files", accept_multiple_files=True)

        # Button to process PDF documents
        process_pdf_button = st.button("Process PDF")

        if pdf_docs and process_pdf_button:
            with st.spinner("Processing PDF..."):
                raw_text = get_pdf_text(pdf_docs)
                text_chunks = get_text_chunks(raw_text)
                get_vector_store(text_chunks)
                st.success("PDF Processing Complete")

        # Input field for user question
        user_question = st.text_input("Ask any question from the PDF Files")

        if user_question:
            model = genai.GenerativeModel('gemini-pro')
            user_input(user_question, model)

if __name__ == "__main__":
    main()
