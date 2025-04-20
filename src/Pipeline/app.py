import streamlit as st
from pipe1 import rag_chain, retriever, format_docs, llm
from evaluation import RAGEvaluator

# Initialize page configuration
st.set_page_config(
    page_title="RAG Chatbot with Evaluation", 
    page_icon="🤖",
    layout="wide"
)

# Initialize the evaluator
evaluator = RAGEvaluator(llm, retriever, format_docs)

# Create two columns for layout
main_col, sidebar_col = st.columns([7, 3])

with main_col:
    st.title("RAG Chatbot with Evaluation")
    st.markdown("""
    Ask questions about the IBA Program Announcement document. 
    The system will retrieve relevant documents, generate an answer, and evaluate its quality.
    """)

# Initialize session state for tracking conversation and evaluation results
if 'conversation_history' not in st.session_state:
    st.session_state.conversation_history = []

if 'evaluation_history' not in st.session_state:
    st.session_state.evaluation_history = []

# Input form for user questions
with main_col:
    with st.form("chat_form"):
        user_input = st.text_input("Ask a question:", placeholder="What programs are offered by the School of Business Studies?")
        col1, col2 = st.columns(2)
        with col1:
            submitted = st.form_submit_button("Generate Response", use_container_width=True)
        with col2:
            clear_history = st.form_submit_button("Clear Chat History", use_container_width=True)

# Clear history if requested
if clear_history:
    st.session_state.conversation_history = []
    st.session_state.evaluation_history = []

# Process the question when submitted
if submitted and user_input:
    # Store and display the question
    st.session_state.conversation_history.append({"role": "user", "content": user_input})
    
    # Show a loading spinner while retrieving documents
    with st.spinner("Retrieving relevant documents..."):
        retrieved_docs = retriever.invoke(user_input)
        formatted_docs = format_docs(retrieved_docs)
    
    # Generate and show the response
    with st.spinner("Generating response..."):  
        response = rag_chain.invoke(user_input)
    
    # Store the response
    st.session_state.conversation_history.append({"role": "assistant", "content": response})
    
    # Evaluate the response
    with st.spinner("Evaluating response quality..."):
        eval_result = evaluator.evaluate_answer(user_input, response)
        st.session_state.evaluation_history.append(eval_result)

# Display conversation history
with main_col:
    st.markdown("### Conversation")
    for message in st.session_state.conversation_history:
        if message["role"] == "user":
            st.markdown(f"**You:** {message['content']}")
        else:
            st.markdown(f"**Assistant:** {message['content']}")
            
    # If we have a new evaluation result, display it
    if submitted and user_input:
        # Create tabs for different displays
        eval_tab, context_tab, metrics_tab = st.tabs(["Evaluation Results", "Retrieved Context", "Performance Metrics"])
        
        with eval_tab:
            st.subheader("Response Quality Metrics")
            eval_metrics = st.columns(3)
            
            with eval_metrics[0]:
                st.metric("Faithfulness", f"{eval_result['evaluation']['faithfulness_score']}/10")
                with st.expander("What does this mean?"):
                    st.write(eval_result['evaluation']['faithfulness_reasoning'])
            
            with eval_metrics[1]:
                st.metric("Relevance", f"{eval_result['evaluation']['relevance_score']}/10")
                with st.expander("What does this mean?"):
                    st.write(eval_result['evaluation']['relevance_reasoning'])
            
            with eval_metrics[2]:
                st.metric("Combined Score", f"{eval_result['evaluation']['combined_score']}/10")
                with st.expander("What does this mean?"):
                    st.write("Combined score is the average of faithfulness and relevance scores.")
            
            # Additional RAGAS metrics if available
            if 'context_relevancy_score' in eval_result['evaluation']:
                add_metrics = st.columns(3)
                with add_metrics[0]:
                    st.metric("Context Relevancy", f"{eval_result['evaluation']['context_relevancy_score']}/10")
                with add_metrics[1]:
                    st.metric("Context Recall", f"{eval_result['evaluation']['context_recall_score']}/10")
                with add_metrics[2]:
                    st.metric("Safety Score", f"{eval_result['evaluation'].get('safety_score', 'N/A')}/10")
        
        with context_tab:
            st.subheader("Retrieved Documents")
            st.text_area("Context used to generate the response:", 
                         value=formatted_docs, 
                         height=300)
            st.caption(f"Number of documents retrieved: {len(retrieved_docs)}")
        
        with metrics_tab:
            st.subheader("System Performance")
            perf_metrics = st.columns(4)
            
            with perf_metrics[0]:
                st.metric("Retrieval Time", f"{eval_result['performance']['retrieval_time']:.3f}s")
            
            with perf_metrics[1]:
                st.metric("Formatting Time", f"{eval_result['performance']['format_time']:.3f}s")
            
            with perf_metrics[2]:
                st.metric("Evaluation Time", f"{eval_result['performance']['evaluation_time']:.3f}s")
            
            with perf_metrics[3]:
                st.metric("Total Time", f"{eval_result['performance']['total_time']:.3f}s")

# Sidebar content
with sidebar_col:
    st.sidebar.header("Evaluation History")
    
    # Display historical scores if available
    if st.session_state.evaluation_history:
        history_data = [
            {
                "Question": st.session_state.conversation_history[i*2]["content"][:30] + "...",
                "Faithfulness": result["evaluation"]["faithfulness_score"],
                "Relevance": result["evaluation"]["relevance_score"],
                "Combined": result["evaluation"]["combined_score"]
            }
            for i, result in enumerate(st.session_state.evaluation_history)
        ]
        
        # Calculate average scores
        avg_faithfulness = sum(item["Faithfulness"] for item in history_data) / len(history_data)
        avg_relevance = sum(item["Relevance"] for item in history_data) / len(history_data)
        avg_combined = sum(item["Combined"] for item in history_data) / len(history_data)
        
        st.sidebar.subheader("Average Scores")
        avg_cols = st.sidebar.columns(3)
        avg_cols[0].metric("Faithfulness", f"{avg_faithfulness:.1f}")
        avg_cols[1].metric("Relevance", f"{avg_relevance:.1f}")
        avg_cols[2].metric("Combined", f"{avg_combined:.1f}")
        
        # Export button
        if st.sidebar.button("Export All Evaluation Results"):
            report_path = evaluator.export_results()
            st.sidebar.success(f"Evaluation results exported to '{report_path}'!")
            
        # Show history table
        st.sidebar.subheader("Query History")
        for i, data in enumerate(history_data):
            with st.sidebar.expander(f"Query {i+1}: {data['Question']}"):
                st.write(f"Faithfulness: {data['Faithfulness']}/10")
                st.write(f"Relevance: {data['Relevance']}/10")
                st.write(f"Combined: {data['Combined']}/10")
    else:
        st.sidebar.info("No evaluation data yet. Ask a question to see results.")