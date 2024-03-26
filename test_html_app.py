import os
import streamlit as st
from PyPDF2 import PdfReader
from dotenv import load_dotenv
import google.generativeai as genai
from langchain.vectorstores import FAISS
from langchain.prompts import PromptTemplate
from streamlit_option_menu import option_menu
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains.question_answering import load_qa_chain
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from htmlTemplates import css, bot_template, user_template



# Load the env file and set the api key for gemini
load_dotenv()
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

def set_page_config():
    st.set_page_config(
        page_title="VBot",
        page_icon="🐶"
    )

def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text

def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
    chunks = text_splitter.split_text(text)
    return chunks

def get_vector_store(text_chunks):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
    vector_store.save_local("faiss_index")

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

def user_input(user_question, model):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    new_db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
    docs = new_db.similarity_search(user_question)
    chain = get_conversational_chain()
    
    response = chain({"input_documents": docs, "question": user_question}, return_only_outputs=True)
    st.write("Reply: ", response["output_text"])

def main():
    set_page_config()
    st.title("🤖 Gemini - ChatBot 🤖")
    st.caption("This bot is made by Tahamid's Group for Info Structure using gemini, streamlit, langchain")

    



    selected = option_menu(
    menu_title=None,  
    options=["ChatBot", "PDF Bot"],
    icons=["robot", "filetype-pdf"],
    orientation="horizontal"
    )

    if selected == "ChatBot":
        model = genai.GenerativeModel('gemini-pro')
        if "chat_session" not in st.session_state:
            st.session_state.chat_session = model.start_chat(history=[])

        col1, col2 = st.columns([5.4, 1])
        with col1:
            user_prompt = st.text_input("Ask any question...")
        with col2:
            if st.button("Clear Chat"):
                st.session_state.chat_session.history = []
                st.rerun()

        

        # Input prompt and send message
        # user_prompt = st.text_input("Ask any question...")

        if user_prompt:
            gemini_response = st.session_state.chat_session.send_message(user_prompt)


        # Include the CSS for chat message styling
        st.markdown(css, unsafe_allow_html=True)

        # Display chat history with the respective templates for user and bot messages
        for message in reversed(st.session_state.chat_session.history):
            # Determine role and select the appropriate template in one line
            html_content = (bot_template if message.role == "model" else user_template).replace("{{MSG}}", message.parts[0].text)
            
            # Output the HTML content to the Streamlit app
            st.markdown(html_content, unsafe_allow_html=True)




        # # Display chat history with avatars
        # for message in reversed(st.session_state.chat_session.history):
        #     role = "assistant" if message.role == "model" else message.role
        #     avatar = "🖥️" if role == "assistant" else "😄"
        #     with st.chat_message(role, avatar=avatar):
        #         st.markdown(message.parts[0].text)
                
    if selected == "PDF Bot":

        pdf_docs = st.file_uploader("Upload your PDF files", accept_multiple_files=True)

        if st.button("Process PDF"):
            if pdf_docs:
                with st.spinner("Processing PDF..."):
                    raw_text = get_pdf_text(pdf_docs)
                    text_chunks = get_text_chunks(raw_text)
                    get_vector_store(text_chunks)
                    st.success("PDF Processing Complete")

        user_question = st.text_input("Ask any question from the PDF Files")
        if user_question:
            model = genai.GenerativeModel('gemini-pro')
            user_input(user_question, model)

        # Display chat history with avatars
        for message in st.session_state.chat_session.history:
            role = "assistant" if message.role == "model" else message.role
            avatar = "🖥️" if role == "assistant" else "😄"
            with st.chat_message(role, avatar=avatar):
                st.markdown(message.parts[0].text)

    # clear chat button
    with st.sidebar:
        if st.button("Clear Conversation", use_container_width=True, type="primary"):
                st.session_state.chat_session.history = []
                st.rerun()

if __name__ == "__main__":
    main()
