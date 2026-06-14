import streamlit as st
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
import tempfile as tmp

from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings

from langchain_groq import ChatGroq
import os
from dotenv import load_dotenv
load_dotenv()


# PAGE CONFIG

st.set_page_config(
    page_title="PDF Quiz Generator",
    page_icon="📘",
    layout="wide",
    initial_sidebar_state="expanded"
)


# CUSTOM CSS (for styling)

st.markdown("""
<style>

body {
    background-color: #f7f7f9;
}

.question-box {
    padding: 20px;
    background: #1e1e1e; /* match dark theme background */
    color: #f5f5f5;       /* light text for visibility */
    border-radius: 12px;
    border: 1px solid #333;
    box-shadow: 0px 2px 6px rgba(0,0,0,0.3);
}

.result-box {
    padding: 15px;
    background: #2b2b2b;
    color: #eaeaea;
    border-left: 5px solid #4a90e2;
    border-radius: 8px;
}

.success-score {
    font-size: 24px;
    font-weight: bold;
    color: #2ecc71;
}

</style>
""", unsafe_allow_html=True)


# PDF EXTRACTION

def extract_pdf_text(uploaded_file):
    with tmp.NamedTemporaryFile(delete=False, suffix=".pdf") as pdf:
        pdf.write(uploaded_file.read())
        pdf_path = pdf.name

    loader = PyPDFLoader(pdf_path)
    document = loader.load()
    return document


# CHUNKING

def split_documents(documents, chunk_size=1000, chunk_overlap=200):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", " ", ""]
    )
    return text_splitter.split_documents(documents)



# CACHED VECTORSTORE

@st.cache_resource
def get_vectorstore(chunks):
    embedding_func = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embedding_func
    )
    return vectorstore


# MAIN APP

def run_app():

    # Sidebar
    st.sidebar.title("📘 PDF Quiz Generator")
    st.sidebar.info("Upload a PDF, enter a topic, and generate a quiz based on the document.")

    st.title("✨ Smart PDF Quiz Generator")
    st.write("Turn any PDF into an interactive quiz using RAG + Groq LLM.")

    uploaded_file = st.file_uploader("📄 Upload your PDF file", type=["pdf"])

    if uploaded_file:
        st.success("PDF uploaded successfully!")

        # Extract
        extracted_text = extract_pdf_text(uploaded_file)
        st.subheader("📌 Text Extraction Complete")

        # Chunk
        chunks = split_documents(extracted_text)
        st.write(f"🔹 Total Chunks: **{len(chunks)}**")

        with st.expander("📘 Preview First Chunk"):
            st.write(chunks[0].page_content[:300])

        # Vector DB
        vectorstore = get_vectorstore(chunks)
        st.success("🧠 Vector Database Created (Temporary & Cached)")

     
        # SESSION STATE
    
        if "context" not in st.session_state:
            st.session_state.context = None

        if "question_number" not in st.session_state:
            st.session_state.question_number = 0

        if "current_question" not in st.session_state:
            st.session_state.current_question = ""

        if "score" not in st.session_state:
            st.session_state.score = 0

        if "asked_questions" not in st.session_state:
            st.session_state.asked_questions = []

    
        # LLM SETUP
      
        groq_api_key = os.getenv("GROQ_API_KEY")

        llm = ChatGroq(
            api_key=groq_api_key,
            model_name="llama-3.3-70b-versatile",
            temperature=0.7,
            max_tokens=1000
        )

        # TOPIC INPUT
      
        topic = st.text_input("🎯 Enter quiz topic")

        if topic and st.session_state.context is None:
            retriever = vectorstore.as_retriever(search_kwargs={"k": 8})
            docs = retriever.invoke(topic)

            st.session_state.context = "\n\n".join([d.page_content for d in docs])
            st.session_state.question_number = 1
            st.session_state.score = 0
            st.session_state.asked_questions = []

     
        # QUIZ LOOP
       
        if 1 <= st.session_state.question_number <= 5:

            st.progress((st.session_state.question_number - 1) / 5)

            QUESTION_TEMPLATE = """
You are a quiz generator.

Generate ONE NEW multiple-choice question.

Context:
{context}

Previously asked questions:
{history}

Rules:
- Do NOT repeat or rephrase previous questions
- Ask something completely new
- 4 options (A, B, C, D)
- DO NOT reveal the correct answer
- DO NOT include explanations

Question number: {q_num}

Format:

Question: <text>
A) <option>
B) <option>
C) <option>
D) <option>
"""

            if not st.session_state.current_question:
                q_prompt = QUESTION_TEMPLATE.format(
                    context=st.session_state.context,
                    history="\n".join(st.session_state.asked_questions),
                    q_num=st.session_state.question_number
                )

                st.session_state.current_question = llm.invoke(q_prompt).content
                st.session_state.asked_questions.append(st.session_state.current_question)

            st.markdown(f"## 📝 Question {st.session_state.question_number}")

            st.markdown(f"""
            <div class="question-box">
            {st.session_state.current_question}
            </div>
            """, unsafe_allow_html=True)

            with st.form(key=f"form_{st.session_state.question_number}"):
                user_answer = st.text_input("Your answer (A/B/C/D)")
                submit = st.form_submit_button("Submit Answer")

            if submit and user_answer:

                EVALUATION_TEMPLATE = """
Evaluate the user's answer using ONLY the context below.

Context:
{context}

Question:
{question}

User Answer:
{user_answer}

Output format:
Result: Correct or Incorrect
Correct Answer: <only if wrong>
Explanation: <1–2 lines based on context>
"""

                eval_prompt = EVALUATION_TEMPLATE.format(
                    context=st.session_state.context,
                    question=st.session_state.current_question,
                    user_answer=user_answer
                )

                evaluation = llm.invoke(eval_prompt).content

                st.markdown(f"""
                <div class="result-box">
                {evaluation}
                </div>
                """, unsafe_allow_html=True)

                if "Result: Correct" in evaluation:
                    st.session_state.score += 1

                st.session_state.current_question = ""
                st.session_state.question_number += 1
                st.rerun()


        # QUIZ COMPLETE
    
        if st.session_state.question_number > 5:
            st.balloons()
            st.markdown("## 🎉 Quiz Completed!")
            st.markdown(f"<p class='success-score'>Your Score: {st.session_state.score} / 5</p>", unsafe_allow_html=True)

            if st.button("🔄 Restart Quiz"):
                st.session_state.context = None
                st.session_state.question_number = 0
                st.session_state.current_question = ""
                st.session_state.score = 0
                st.session_state.asked_questions = []
                st.rerun()


run_app()
