import streamlit as st
import requests
import time

# ----------------------------------
# Configuration
# ----------------------------------
FASTAPI_URL = "http://127.0.0.1:8000"
QUERY_ENDPOINT = f"{FASTAPI_URL}/api/v1/query"
UPLOAD_ENDPOINT = f"{FASTAPI_URL}/upload/pdf"

# ----------------------------------
# Page Configuration
# ----------------------------------
st.set_page_config(page_title="Regulatory Compliance RAG Assistant", layout="wide")
st.title("***** Regulatory Compliance RAG Assistant *****")
st.write("""
    Ask questions from uploaded RBI / SEBI / Basel
    regulatory documents.
    """)

# ----------------------------------
# Session State
# ----------------------------------
if "messages" not in st.session_state:
    st.session_state.messages = []


# ----------------------------------
# Sidebar
# ----------------------------------
with st.sidebar:
    st.header("Document Upload")
    uploaded_file = st.file_uploader("Upload PDF", type=["pdf"])
    if uploaded_file:
        if st.button("Process Document"):
            with st.spinner("Uploading and indexing document..."):
                files = {"file": (uploaded_file.name, uploaded_file, "application/pdf")}
                response = requests.post(UPLOAD_ENDPOINT, files=files)
                if response.status_code in [200, 201]:
                    st.success("Document processed successfully")
                else:
                    st.error(response.text)


# ----------------------------------
# Display Chat History
# ----------------------------------
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])


# ----------------------------------
# User Query
# ----------------------------------
question = st.chat_input("Ask your compliance question...")
if question:
    # User message
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.write(question)
    # Assistant response
    with st.chat_message("assistant"):
        with st.spinner("Searching documents..."):
            start = time.time()
            payload = {"question": question, "chat_history": st.session_state.messages}
            response = requests.post(QUERY_ENDPOINT, json=payload)
            latency = round((time.time() - start) * 1000, 2)
            if response.status_code == 200:
                result = response.json()
                answer = result.get("answer", "")
                st.write(answer)
                st.session_state.messages.append(
                    {"role": "assistant", "content": answer}
                )
                st.divider()

                # ----------------------------------
                # Metadata
                # Show only for RAG / Regulatory queries
                # ----------------------------------
                if result.get("query_type") == "rag" and result.get("tool_used"):
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Tool Used", result.get("tool_used"))
                    with col2:
                        st.metric("Confidence", result.get("confidence"))
                    with col3:
                        st.metric("Latency", f"{latency} ms")
                # ----------------------------------
                # Citations
                # ----------------------------------
                sources = result.get("sources", [])
                if sources:
                    st.subheader("Document Citations")
                    for idx, source in enumerate(sources, start=1):
                        title = (
                            f"Source {idx} - " f"{source.get('file_name','Unknown')}"
                        )
                        with st.expander(title):
                            st.write("Document ID:", source.get("document_id"))
                            st.write("Page Number:", source.get("page_number"))
                            st.write("Section:", source.get("section_number"))
                            st.write("Regulation Type:", source.get("regulation_type"))
                            st.write(
                                "Retrieval Method:", source.get("retrieval_method")
                            )
                            # Scores
                            st.write("Vector Score:", source.get("vector_score"))
                            st.write("FTS Score:", source.get("fts_score"))
                            st.write("Hybrid Score:", source.get("hybrid_score"))
                            st.divider()
                            st.write("Text Snippet:")
                            st.info(source.get("snippet"))
            else:
                st.error(response.text)
