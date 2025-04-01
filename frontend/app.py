import streamlit as st
import requests

# Backend API URL
BACKEND_URL = "http://backend:8000"  # Use service name in Docker

st.title("Black's Law Dictionary RAG")

# Index the dictionary (run once)
if 'indexed' not in st.session_state:
    st.session_state.indexed = False

if not st.session_state.indexed:
    if st.button("Index Dictionary"):
        with st.spinner("Indexing..."):
            response = requests.post(f"{BACKEND_URL}/index")
            if response.status_code == 200:
                st.success(response.json()["message"])
                st.session_state.indexed = True
            else:
                st.error("Failed to index dictionary")

# Query section
st.header("Ask a Legal Question")
question = st.text_input("Enter your question", placeholder="e.g., What's an affidavit?")
if st.button("Submit Query"):
    if question:
        with st.spinner("Fetching answer..."):
            response = requests.post(f"{BACKEND_URL}/query", json={"question": question})
            if response.status_code == 200:
                st.write(f"**Answer:** {response.json()['answer']}")
            else:
                st.error("Error fetching answer")
    else:
        st.warning("Please enter a question")

# Lookup section
st.header("Dictionary Lookup")
term = st.text_input("Enter a term", placeholder="e.g., AFFIDAVIT")
if st.button("Search Term"):
    if term:
        with st.spinner("Searching..."):
            response = requests.post(f"{BACKEND_URL}/lookup", json={"term": term})
            if response.status_code == 200:
                st.write(f"**Definition:** {response.json()['definition']}")
            else:
                st.error(response.json().get("detail", "Error fetching definition"))
    else:
        st.warning("Please enter a term")