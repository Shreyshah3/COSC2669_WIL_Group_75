import streamlit as st
import streamlit.components.v1 as components

from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_chroma import Chroma



# PAGE CONFIGURATION


st.set_page_config(
    page_title="Banking PDF RAG Chatbot",
    page_icon="🏦",
    layout="centered"
)



# TITLE


st.title("🏦 Banking PDF RAG Chatbot")

st.write(
    "Ask questions about the banking FAQ knowledge base."
)

st.caption(
    "Powered by Ollama + LangChain + ChromaDB + RAG"
)



# CLEAR CHAT


if st.button("Clear chat"):

    st.session_state.messages = []

    st.rerun()



# LOAD EMBEDDING MODEL


@st.cache_resource
def load_embeddings():

    return OllamaEmbeddings(
        model="nomic-embed-text"
    )



# LOAD VECTOR DATABASE


@st.cache_resource
def load_vectorstore():

    embeddings = load_embeddings()

    return Chroma(
        collection_name="banking_faq",
        embedding_function=embeddings,
        persist_directory="vectorstore/banking_faq"
    )



# LOAD LLM


@st.cache_resource
def load_llm():

    return ChatOllama(
        model="llama3.2",
        temperature=0
    )



# INITIALIZE COMPONENTS


vectorstore = load_vectorstore()

llm = load_llm()

retriever = vectorstore.as_retriever(
    search_kwargs={
        "k": 4
    }
)



# CHAT HISTORY


if "messages" not in st.session_state:

    st.session_state.messages = []



# DISPLAY CHAT HISTORY


for index, message in enumerate(st.session_state.messages):

    with st.chat_message(message["role"]):

        st.markdown(message["content"])

        # Add Copy button only to assistant messages
        if message["role"] == "assistant":

            components.html(
                f"""
                <button
                    onclick="navigator.clipboard.writeText({message['content']!r})"
                    style="
                        padding: 5px 12px;
                        border: 1px solid #ccc;
                        border-radius: 6px;
                        background: white;
                        cursor: pointer;
                        font-size: 13px;
                    "
                >
                    📋 Copy
                </button>
                """,
                height=40
            )



# USER INPUT


question = st.chat_input(
    "Ask a banking question..."
)


if question:

   
    # Display user question
 

    st.session_state.messages.append(
        {
            "role": "user",
            "content": question
        }
    )

    with st.chat_message("user"):

        st.markdown(question)


    
    # Retrieve relevant FAQs
    

    retrieved_docs = retriever.invoke(
        question
    )


    
    # Build context
    

    context = "\n\n".join(
        doc.page_content
        for doc in retrieved_docs
    )


    
    # Build RAG prompt
    

    prompt = f"""
You are a helpful banking FAQ assistant.

Answer the user's question using ONLY the
information provided in the context.

If the answer cannot be found in the context,
say:

"I could not find this information in the banking FAQ."

Do not invent information.
Do not use outside knowledge.

Context:
{context}

User question:
{question}

Answer:
"""


    
    # Generate answer
    

    with st.chat_message("assistant"):

        with st.spinner("Searching the banking FAQ..."):

            response = llm.invoke(prompt)

            answer = response.content

        st.markdown(answer)

        
        # COPY BUTTON
        

        components.html(
            f"""
            <button
                onclick="navigator.clipboard.writeText({answer!r})"
                style="
                    padding: 5px 12px;
                    border: 1px solid #ccc;
                    border-radius: 6px;
                    background: white;
                    cursor: pointer;
                    font-size: 13px;
                "
            >
                📋 Copy
            </button>
            """,
            height=40
        )


    
    # Save assistant response
    

    st.session_state.messages.append(
        {
            "role": "assistant",
            "content": answer
        }
    )