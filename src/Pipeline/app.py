import streamlit as st
from pipe1 import rag_chain, retriever, format_docs  # Ensure RAG chain supports invocation

st.set_page_config(page_title="RAG Chatbot")

st.title("RAG Chatbot")

with st.form("chat_form"):
    user_input = st.text_input("Ask a question:")
    submitted = st.form_submit_button("Generate Response")

if submitted and user_input:
    st.subheader("Retrieved Documents:") # Show the retrieved documents
    st.write((retriever | format_docs) .invoke(user_input))
    
    st.subheader("Response:") # Show the generated response
    with st.spinner("Generating response..."):  # Show a spinner while generating the response
        response = rag_chain.invoke(user_input)
    
    st.write(response) 
