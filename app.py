import streamlit as st
from dotenv import load_dotenv
from PyPDF2 import PdfReader

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate


def get_pdf_text(pdf_docs):
    text = ""

    for pdf in pdf_docs:
        for page in PdfReader(pdf).pages:
            text += page.extract_text() or ""

    return text


def get_text_chunks(text):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )

    return splitter.split_text(text)


def get_rag_components(chunks):

    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    vectorstore = FAISS.from_texts(chunks, embeddings)

    retriever = vectorstore.as_retriever(
        search_kwargs={"k": 4}
    )

    llm = ChatGroq(
        model="openai/gpt-oss-20b",
        temperature=0.2
    )

    prompt = ChatPromptTemplate.from_template("""
You are an assistant answering questions about the user's documents.

Use the following context to answer the question.

If the answer cannot be found in the context, say:
"I couldn't find the answer in the provided documents."

Do not make up information.

Context:
{context}

Question:
{question}
""")

    return llm, retriever, prompt


def answer_question(question):

    docs = st.session_state.retriever.invoke(question)

    context = "\n\n".join(
        doc.page_content for doc in docs
    )

    messages = st.session_state.messages + [
        ("human", st.session_state.prompt.format(
            context=context,
            question=question
        ))
    ]

    response = st.session_state.llm.invoke(messages)

    st.session_state.messages.extend([
        ("human", question),
        ("ai", response.content)
    ])


def main():

    load_dotenv()

    st.set_page_config(
        page_title="Chat with PDFs",
        page_icon="📚"
    )

    if "retriever" not in st.session_state:
        st.session_state.retriever = None

    if "llm" not in st.session_state:
        st.session_state.llm = None

    if "prompt" not in st.session_state:
        st.session_state.prompt = None

    if "messages" not in st.session_state:
        st.session_state.messages = []

    st.header("Chat with Multiple PDFs 📚")

    for role, message in st.session_state.messages:
        st.chat_message(
            "user" if role == "human" else "assistant"
        ).write(message)

    question = st.chat_input(
        "Ask a question about your documents:"
    )

    if question:

        if st.session_state.retriever is None:
            st.warning("Please upload and process your PDFs first.")

        else:
            answer_question(question)

            st.rerun()

    with st.sidebar:

        st.subheader("Your documents")

        pdf_docs = st.file_uploader(
            "Upload your PDFs",
            accept_multiple_files=True,
            type="pdf"
        )

        if st.button("Process") and pdf_docs:

            with st.spinner("Processing..."):

                text = get_pdf_text(pdf_docs)
                if not text.strip():
                    st.error("Could not extract any text from the PDF.")
                    st.stop()
                chunks = get_text_chunks(text)

                if not chunks:
                    st.error("No text chunks were created.")
                    st.stop()

                llm, retriever, prompt = get_rag_components(chunks)

                st.session_state.llm = llm
                st.session_state.retriever = retriever
                st.session_state.prompt = prompt
                st.session_state.messages = []

            st.success(f"Processed {len(chunks)} chunks!")


if __name__ == "__main__":
    main()