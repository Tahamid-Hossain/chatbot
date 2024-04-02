import os
import textwrap
from PIL import Image
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
        page_title="Gemini Bot",
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

def get_gemini_response(input,image):
    model = genai.GenerativeModel('gemini-pro-vision')
    if input!="":
       response = model.generate_content([input,image])
    else:
       response = model.generate_content(image)
    return response.text


def chat_func(prompt, img):
    model = genai.GenerativeModel('gemini-pro-vision')

    try:
        response = model.generate_content([prompt, img])
        return response.text
    except Exception as e:
        return f"An error occured: {e}. Please try again."


def convert_image_to_pil(st_image):
    import io
    from PIL import Image
    image_data = st_image.read()
    pil_image = Image.open(io.BytesIO(image_data))
    return pil_image


def main():
    set_page_config()
    st.title("🤖 Gemini Bot 🤖")
    st.caption("This bot is made by Tahamid's Group for Info Structure using gemini, streamlit, langchain")

 
    selected = option_menu(
    menu_title=None,  
    options=["ChatBot", "PDF Bot", "Vision Bot"],
    icons=["robot", "filetype-pdf", "image"],
    orientation="horizontal"
)

    if selected == "PDF Bot":
        pdf_docs = st.file_uploader("Upload your PDF files & Click Process Button", accept_multiple_files=True)

        if st.button("Process PDF"):
            if pdf_docs:
                with st.spinner("Processing PDF..."):
                    raw_text = get_pdf_text(pdf_docs)
                    text_chunks = get_text_chunks(raw_text)
                    get_vector_store(text_chunks)
                    st.success("PDF Processing Complete")
                    
        # Revised chat interface for PDF Bot
        if "pdf_chat_session" not in st.session_state:
            st.session_state.pdf_chat_session = []

        col1, col2 = st.columns([5.4, 1])
        with col1:
            user_question = st.chat_input("Ask any question from the PDF Files")
        with col2:
            if st.button("Clear Chat"):
                st.session_state.pdf_chat_session = []
                st.rerun()

        if user_question:
            model = genai.GenerativeModel('gemini-pro')
            # This is where the modification is significant.
            # Instead of directly writing the response, we append it to a session state list.
            response = user_input(user_question, model)
            
            # Append user question and response to the session state for rendering as chat
            st.session_state.pdf_chat_session.append({"role": "user", "text": user_question})
            st.session_state.pdf_chat_session.append({"role": "bot", "text": response})
            
            # Render chat
            st.markdown(css, unsafe_allow_html=True)  # Assuming CSS is defined for chat appearance
            for message in reversed(st.session_state.pdf_chat_session):
                # Choose the template based on the message role
                html_content = (bot_template if message["role"] == "bot" else user_template).replace("{{MSG}}", message["text"])
                st.markdown(html_content, unsafe_allow_html=True)



    if selected == "Vision Bot":
        img = st.file_uploader("Upload an image", type=['png', 'jpg', 'jpeg','gif'])

        if img:
            st.image(img, caption='Chat with this image')

            prompt = st.chat_input('Ask a question about your image:')

            if prompt:
                pil_image = convert_image_to_pil(img)

                with st.spinner('Analyzing your image ...'):
                    answer = chat_func(prompt, pil_image)
                    st.text_area('Gemini Answer: ', value=answer)


                if 'history' not in st.session_state:
                    st.session_state.history = 'Chat History\n'

                value = f'**Question**: {prompt}: \n\n **Answer**: {answer}'
                st.session_state.history = f'{value} \n\n {"-" * 100} \n\n {st.session_state.history}'

                # h = st.session_state.history
                # st.text_area(label='Chat History:', value=h, height=800, key='history')

if __name__ == "__main__":
    main()
