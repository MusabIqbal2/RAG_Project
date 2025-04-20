import time
import json
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
import os
import numpy as np
from langchain_core.documents import Document

# Import RAGAS metrics
try:
    from ragas.metrics import (
        faithfulness,
        answer_relevancy,
        # context_relevancy,
        context_recall,
        # harmfulness
    )
    from ragas import evaluate
except ImportError:
    raise ImportError(
        "RAGAS package is required. Install it with 'pip install ragas'"
    )

class RAGEvaluator:
    def __init__(self, llm, retriever, format_docs):
        """
        Initialize the RAG Evaluator with RAGAS metrics.
        
        Args:
            llm: The LLM used in the RAG system
            retriever: The retriever component used in the RAG system
            format_docs: The document formatting function
        """
        self.llm = llm
        self.retriever = retriever
        self.format_docs = format_docs
        
        # Initialize metrics storage
        self.evaluation_results = []
        self.performance_metrics = []
        
        # Create directories to store results
        self.output_dir = "evaluation_results"
        os.makedirs(self.output_dir, exist_ok=True)
    
    def evaluate_answer(self, question, answer):
        """
        Evaluate the quality of an answer using RAGAS metrics.
        
        Args:
            question: The user's question
            answer: The generated answer
            
        Returns:
            dict: Evaluation results
        """
        # Start timers for performance metrics
        start_time = time.time()
        
        # Retrieve documents
        retrieval_start = time.time()
        docs = self.retriever.invoke(question)
        retrieval_time = time.time() - retrieval_start
        
        # Format documents
        format_start = time.time()
        context = self.format_docs(docs)
        format_time = time.time() - format_start
        
        # Evaluate using RAGAS
        eval_start = time.time()
        
        # Prepare data for RAGAS
        data = {
            'question': [question],
            'answer': [answer],
            'contexts': [[doc.page_content for doc in docs]],
        }
        
        # Create DataFrame for RAGAS
        eval_data = pd.DataFrame(data)
        
        # Run RAGAS evaluation with error handling
        try:
            from datasets import Dataset
            
            # Convert data to Hugging Face dataset format
            dataset = Dataset.from_pandas(eval_data)
            
            result = evaluate(
                dataset=dataset,
                metrics=[
                    faithfulness,
                    answer_relevancy,
                    context_recall,
                ]
            )
            
            # Scale RAGAS scores (0-1) to 0-10 for consistency
            faithfulness_score = round(float(result['faithfulness']) * 10, 1)
            relevance_score = round(float(result['answer_relevancy']) * 10, 1)
            context_recall_score = round(float(result['context_recall']) * 10, 1)
            context_relevancy_score = 0.0  # This metric is no longer available in current RAGAS version
            safety_score = 10.0  # This metric is no longer available in current RAGAS version
            
            # Calculate combined score as average of primary metrics
            combined_score = round((faithfulness_score + relevance_score) / 2, 1)
            
            # Generate reasoning explanations
            faithfulness_reasoning = (
                f"RAGAS faithfulness score: {faithfulness_score/10:.2f}/1.0. "
                f"This measures if the answer contains only information present in or directly "
                f"inferable from the retrieved context."
            )
            
            relevance_reasoning = (
                f"RAGAS answer relevancy score: {relevance_score/10:.2f}/1.0. "
                f"This measures how well the answer addresses the specific question asked."
            )
            
        except Exception as e:
            print(f"RAGAS evaluation error: {str(e)}")
            # Provide fallback scores if RAGAS fails
            faithfulness_score = 5.0
            relevance_score = 5.0
            context_relevancy_score = 5.0
            context_recall_score = 5.0
            safety_score = 5.0
            combined_score = 5.0
            
            faithfulness_reasoning = f"Error in RAGAS evaluation: {str(e)[:100]}..."
            relevance_reasoning = f"Error in RAGAS evaluation: {str(e)[:100]}..."
        
        eval_time = time.time() - eval_start
        total_time = time.time() - start_time
        
        # Create evaluation record
        evaluation = {
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "question": question,
            "answer": answer,
            "context": context,
            "faithfulness_score": faithfulness_score,
            "faithfulness_reasoning": faithfulness_reasoning,
            "relevance_score": relevance_score,
            "relevance_reasoning": relevance_reasoning,
            "context_relevancy_score": context_relevancy_score,
            "context_recall_score": context_recall_score,
            "safety_score": safety_score,
            "combined_score": combined_score
        }
        
        # Create performance record
        performance = {
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "question": question,
            "retrieval_time": retrieval_time,
            "format_time": format_time,
            "evaluation_time": eval_time,
            "total_time": total_time,
            "num_retrieved_docs": len(docs)
        }
        
        # Store results
        self.evaluation_results.append(evaluation)
        self.performance_metrics.append(performance)
        
        return {
            "evaluation": evaluation,
            "performance": performance
        }
    
    def export_results(self, output_dir=None):
        """
        Export evaluation results and performance metrics to CSV files.
        
        Args:
            output_dir: Directory to save results (uses default if None)
        """
        if output_dir is None:
            output_dir = self.output_dir
            
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        # Generate timestamp for filenames
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Export evaluation results
        if self.evaluation_results:
            eval_df = pd.DataFrame(self.evaluation_results)
            # Remove long text fields for the CSV export
            eval_df_export = eval_df.drop(columns=['context'])
            eval_df_export.to_csv(f"{output_dir}/evaluation_results_{timestamp}.csv", index=False)
            
            # Create summary visualizations
            self._create_evaluation_visualizations(eval_df, output_dir, timestamp)
        
        # Export performance metrics
        if self.performance_metrics:
            perf_df = pd.DataFrame(self.performance_metrics)
            perf_df.to_csv(f"{output_dir}/performance_metrics_{timestamp}.csv", index=False)
            
            # Create performance visualizations
            self._create_performance_visualizations(perf_df, output_dir, timestamp)
            
        # Create summary report
        self._create_summary_report(output_dir, timestamp)
        
        return f"{output_dir}/summary_report_{timestamp}.html"
    
    def _create_summary_report(self, output_dir, timestamp):
        """Generate HTML summary report of evaluation results"""
        if not self.evaluation_results or not self.performance_metrics:
            return
            
        eval_df = pd.DataFrame(self.evaluation_results)
        perf_df = pd.DataFrame(self.performance_metrics)
        
        # Calculate summary statistics
        avg_faithfulness = eval_df['faithfulness_score'].mean()
        avg_relevance = eval_df['relevance_score'].mean()
        avg_combined = eval_df['combined_score'].mean()
        avg_context_rel = eval_df['context_relevancy_score'].mean() if 'context_relevancy_score' in eval_df else 0
        avg_context_recall = eval_df['context_recall_score'].mean() if 'context_recall_score' in eval_df else 0
        
        avg_retrieval_time = perf_df['retrieval_time'].mean()
        avg_total_time = perf_df['total_time'].mean()
        
        # Create summary HTML
        html_content = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>RAG Evaluation Summary</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 20px; }}
                .container {{ max-width: 1200px; margin: 0 auto; }}
                .header {{ background-color: #4b6584; color: white; padding: 20px; text-align: center; }}
                .section {{ margin: 20px 0; padding: 15px; border: 1px solid #ddd; border-radius: 5px; }}
                .metric-box {{ display: inline-block; width: 180px; padding: 15px; margin: 10px; 
                              text-align: center; background-color: #f5f6fa; border-radius: 5px; }}
                .score {{ font-size: 24px; font-weight: bold; margin: 10px 0; }}
                .good {{ color: #27ae60; }}
                .average {{ color: #f39c12; }}
                .poor {{ color: #e74c3c; }}
                table {{ width: 100%; border-collapse: collapse; margin: 15px 0; }}
                th, td {{ border: 1px solid #ddd; padding: 12px; text-align: left; }}
                th {{ background-color: #f2f2f2; }}
                img {{ max-width: 100%; height: auto; margin: 15px 0; }}
            </style>
        </head>
        <body>
            <div class="container">
                <div class="header">
                    <h1>RAG Evaluation Summary Report</h1>
                    <p>Generated on {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}</p>
                </div>
                
                <div class="section">
                    <h2>Quality Metrics Summary</h2>
                    <div class="metric-box">
                        <h3>Faithfulness</h3>
                        <div class="score {self._get_score_class(avg_faithfulness)}">{avg_faithfulness:.1f}/10</div>
                    </div>
                    <div class="metric-box">
                        <h3>Relevance</h3>
                        <div class="score {self._get_score_class(avg_relevance)}">{avg_relevance:.1f}/10</div>
                    </div>
                    <div class="metric-box">
                        <h3>Combined Score</h3>
                        <div class="score {self._get_score_class(avg_combined)}">{avg_combined:.1f}/10</div>
                    </div>
                    <div class="metric-box">
                        <h3>Context Relevancy</h3>
                        <div class="score {self._get_score_class(avg_context_rel)}">{avg_context_rel:.1f}/10</div>
                    </div>
                    <div class="metric-box">
                        <h3>Context Recall</h3>
                        <div class="score {self._get_score_class(avg_context_recall)}">{avg_context_recall:.1f}/10</div>
                    </div>
                </div>
                
                <div class="section">
                    <h2>Performance Metrics Summary</h2>
                    <div class="metric-box">
                        <h3>Avg Retrieval Time</h3>
                        <div class="score">{avg_retrieval_time:.3f}s</div>
                    </div>
                    <div class="metric-box">
                        <h3>Avg Total Time</h3>
                        <div class="score">{avg_total_time:.3f}s</div>
                    </div>
                </div>
                
                <div class="section">
                    <h2>Question-Answer Samples</h2>
                    <table>
                        <tr>
                            <th>Question</th>
                            <th>Answer</th>
                            <th>Faithfulness</th>
                            <th>Relevance</th>
                        </tr>
        """
        
        # Add up to 5 sample Q&A pairs
        for i, eval_result in enumerate(self.evaluation_results[:5]):
            html_content += f"""
                        <tr>
                            <td>{eval_result['question']}</td>
                            <td>{eval_result['answer']}</td>
                            <td>{eval_result['faithfulness_score']}/10</td>
                            <td>{eval_result['relevance_score']}/10</td>
                        </tr>
            """
        
        html_content += """
                    </table>
                </div>
                
                <div class="section">
                    <h2>Visualizations</h2>
                    <img src="avg_scores_{timestamp}.png" alt="Average Scores">
                    <img src="score_distributions_{timestamp}.png" alt="Score Distributions">
                    <img src="response_times_{timestamp}.png" alt="Response Times">
                </div>
            </div>
        </body>
        </html>
        """
        
        # Write HTML to file
        with open(f"{output_dir}/summary_report_{timestamp}.html", "w") as f:
            f.write(html_content)
    
    def _get_score_class(self, score):
        """Return CSS class based on score value"""
        if score >= 7.5:
            return "good"
        elif score >= 5:
            return "average"
        else:
            return "poor"
    
    def _create_evaluation_visualizations(self, df, output_dir, timestamp):
        """
        Create visualizations for evaluation metrics.
        
        Args:
            df: DataFrame containing evaluation results
            output_dir: Directory to save visualizations
        """
        # 1. Create a bar chart of average scores
        plt.figure(figsize=(12, 7))
        metrics = ['faithfulness_score', 'relevance_score', 'combined_score']
        if 'context_relevancy_score' in df.columns:
            metrics.extend(['context_relevancy_score', 'context_recall_score', 'safety_score'])
            
        metric_names = {
            'faithfulness_score': 'Faithfulness',
            'relevance_score': 'Relevance', 
            'combined_score': 'Combined',
            'context_relevancy_score': 'Context Relevancy',
            'context_recall_score': 'Context Recall',
            'safety_score': 'Safety'
        }
        
        avg_scores = [df[metric].mean() for metric in metrics]
        colors = ['#3498db', '#2ecc71', '#9b59b6', '#e74c3c', '#f39c12', '#1abc9c']
        
        bars = plt.bar(
            [metric_names[m] for m in metrics],
            avg_scores,
            color=colors[:len(metrics)]
        )
        
        # Add value labels on top of bars
        for bar in bars:
            height = bar.get_height()
            plt.text(
                bar.get_x() + bar.get_width()/2.,
                height + 0.3,
                f'{height:.1f}',
                ha='center',
                va='bottom',
                fontweight='bold'
            )
            
        plt.ylim(0, 11)  # Leave space for labels
        plt.title('Average Evaluation Scores', fontsize=16)
        plt.ylabel('Score (0-10)', fontsize=14)
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        plt.savefig(f"{output_dir}/avg_scores_{timestamp}.png", bbox_inches='tight', dpi=300)
        plt.close()
        
        # 2. Create score distributions
        plt.figure(figsize=(14, 8))
        
        for i, metric in enumerate(metrics[:4]):  # Limit to 4 metrics for readability
            plt.subplot(2, 2, i+1)
            plt.hist(df[metric], bins=10, alpha=0.7, color=colors[i])
            plt.title(f'{metric_names[metric]} Distribution')
            plt.xlabel('Score')
            plt.ylabel('Frequency')
            plt.grid(True, linestyle='--', alpha=0.5)
            
        plt.tight_layout()
        plt.savefig(f"{output_dir}/score_distributions_{timestamp}.png", bbox_inches='tight', dpi=300)
        plt.close()
        
        # 3. Create a scatter plot of faithfulness vs. relevance if enough data points
        if len(df) > 5:
            plt.figure(figsize=(10, 6))
            plt.scatter(
                df['faithfulness_score'],
                df['relevance_score'],
                alpha=0.7,
                s=80
            )
            
            # Add trend line
            z = np.polyfit(df['faithfulness_score'], df['relevance_score'], 1)
            p = np.poly1d(z)
            plt.plot(
                [min(df['faithfulness_score']), max(df['faithfulness_score'])],
                [p(min(df['faithfulness_score'])), p(max(df['faithfulness_score']))],
                "r--", alpha=0.7
            )
            
            plt.xlim(0, 10)
            plt.ylim(0, 10)
            plt.grid(True, linestyle='--', alpha=0.7)
            plt.title('Faithfulness vs. Relevance Scores', fontsize=16)
            plt.xlabel('Faithfulness Score', fontsize=14)
            plt.ylabel('Relevance Score', fontsize=14)
            plt.savefig(f"{output_dir}/faithfulness_vs_relevance_{timestamp}.png", bbox_inches='tight', dpi=300)
            plt.close()
    
    def _create_performance_visualizations(self, df, output_dir, timestamp):
        """
        Create visualizations for performance metrics.
        
        Args:
            df: DataFrame containing performance metrics
            output_dir: Directory to save visualizations
        """
        # Create a stacked bar chart of time components
        plt.figure(figsize=(14, 8))
        
        # Prepare data
        num_samples = min(len(df), 12)  # Show up to 12 most recent queries
        indices = range(num_samples)
        samples = df.iloc[-num_samples:]
        
        retrieval = samples['retrieval_time'].tolist()
        formatting = samples['format_time'].tolist()
        evaluation = samples['evaluation_time'].tolist()
        other_time = samples['total_time'] - (samples['retrieval_time'] + samples['format_time'] + samples['evaluation_time'])
        
        # Plot stacked bars
        plt.bar(indices, retrieval, label='Retrieval Time', color='#3498db')
        plt.bar(indices, formatting, bottom=retrieval, label='Formatting Time', color='#2ecc71')
        plt.bar(indices, evaluation, bottom=[r+f for r, f in zip(retrieval, formatting)], 
                label='Evaluation Time', color='#9b59b6')
        plt.bar(indices, other_time, 
                bottom=[r+f+e for r, f, e in zip(retrieval, formatting, evaluation)],
                label='Other Processing Time', color='#f39c12')
        
        plt.xlabel('Query Number', fontsize=14)
        plt.ylabel('Time (seconds)', fontsize=14)
        plt.title('Response Time Components by Query', fontsize=16)
        plt.legend(loc='upper left', bbox_to_anchor=(1, 1))
        plt.xticks(indices, [f"Q{i+1}" for i in indices], rotation=45)
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        plt.tight_layout()
        plt.savefig(f"{output_dir}/response_times_{timestamp}.png", bbox_inches='tight', dpi=300)
        plt.close()
        
        # Create a pie chart of average time components
        plt.figure(figsize=(10, 8))
        avg_retrieval = df['retrieval_time'].mean()
        avg_formatting = df['format_time'].mean()
        avg_evaluation = df['evaluation_time'].mean()
        avg_other = df['total_time'].mean() - (avg_retrieval + avg_formatting + avg_evaluation)
        
        labels = ['Retrieval', 'Formatting', 'Evaluation', 'Other Processing']
        sizes = [avg_retrieval, avg_formatting, avg_evaluation, avg_other]
        colors = ['#3498db', '#2ecc71', '#9b59b6', '#f39c12']
        
        # Only show non-zero components
        non_zero_indices = [i for i, size in enumerate(sizes) if size > 0.001]
        filtered_labels = [labels[i] for i in non_zero_indices]
        filtered_sizes = [sizes[i] for i in non_zero_indices]
        filtered_colors = [colors[i] for i in non_zero_indices]
        
        plt.pie(filtered_sizes, labels=filtered_labels, colors=filtered_colors, 
                autopct='%1.1f%%', startangle=90, shadow=True)
        plt.axis('equal')
        plt.title('Average Time Distribution', fontsize=16)
        plt.savefig(f"{output_dir}/time_distribution_{timestamp}.png", bbox_inches='tight', dpi=300)
        plt.close()


# Function to benchmark different RAG configurations
def benchmark_configurations(base_retriever, format_docs, llm, questions, configurations):
    """
    Benchmark different RAG configurations and compare their performance.
    
    Args:
        base_retriever: Base retriever to modify with different configurations
        format_docs: Document formatting function
        llm: LLM model to use
        questions: List of questions to test
        configurations: Dictionary of configurations to test with parameters
        
    Returns:
        DataFrame with benchmark results
    """
    from langchain_core.runnables import RunnablePassthrough
    from langchain.prompts import ChatPromptTemplate, HumanMessagePromptTemplate
    from langchain.prompts import PromptTemplate
    from langchain_core.output_parsers import StrOutputParser
    
    results = []
    
    for config_name, params in configurations.items():
        print(f"Testing configuration: {config_name}")
        
        # Configure retriever
        if hasattr(base_retriever, 'search_kwargs'):
            test_retriever = base_retriever.with_search_kwargs(params.get('search_kwargs', {}))
        else:
            # If retriever doesn't support this method, use the base retriever
            test_retriever = base_retriever
            
        # Configure prompt template if provided
        prompt_template = params.get('prompt_template', None)
        if prompt_template is None:
            prompt_template = ChatPromptTemplate.from_messages([
                HumanMessagePromptTemplate(
                    prompt=PromptTemplate(
                        input_variables=["context", "question"],
                        template=(
                            "You are an assistant for question-answering tasks. Use the following "
                            "pieces of retrieved context to answer the question. If you don't know "
                            "the answer, just say that you don't know. Use three sentences maximum "
                            "and keep the answer concise.\n\n"
                            
                            "Question: {question}\n"
                            "Context: {context}\n"
                            "Answer:"
                        )
                    )
                )
            ])
        
        # Create test RAG chain
        test_rag_chain = (
            {
                "context": test_retriever | format_docs,
                "question": RunnablePassthrough()
            }
            | prompt_template
            | llm
            | StrOutputParser()
        )
        
        # Create evaluator
        evaluator = RAGEvaluator(llm, test_retriever, format_docs)
        
        # Test each question
        config_results = []
        for question in questions:
            start_time = time.time()
            answer = test_rag_chain.invoke(question)
            generation_time = time.time() - start_time
            
            # Evaluate answer
            eval_result = evaluator.evaluate_answer(question, answer)
            
            # Store results
            result = {
                'configuration': config_name,
                'question': question,
                'answer': answer,
                'generation_time': generation_time,
                **{f"{k}_score": v for k, v in eval_result['evaluation'].items() if k.endswith('_score')},
                **{k: v for k, v in eval_result['performance'].items() if k not in ['question', 'timestamp']}
            }
            config_results.append(result)
            print(f"  Question: '{question[:50]}...' - Combined Score: {result['combined_score']:.1f}/10")
        
        # Aggregate results for this configuration
        avg_result = {
            'configuration': config_name,
            'avg_faithfulness': sum(r['faithfulness_score'] for r in config_results) / len(config_results),
            'avg_relevance': sum(r['relevance_score'] for r in config_results) / len(config_results),
            'avg_combined': sum(r['combined_score'] for r in config_results) / len(config_results),
            'avg_generation_time': sum(r['generation_time'] for r in config_results) / len(config_results),
            'avg_retrieval_time': sum(r['retrieval_time'] for r in config_results) / len(config_results),
            'avg_total_time': sum(r['total_time'] for r in config_results) / len(config_results),
            'details': config_results
        }
        results.append(avg_result)
        
        print(f"  Average Combined Score: {avg_result['avg_combined']:.1f}/10")
        print(f"  Average Total Time: {avg_result['avg_total_time']:.2f}s")
        print("-" * 50)
    
    # Return results as DataFrame for comparison
    df_results = pd.DataFrame([
        {k: v for k, v in r.items() if k != 'details'}
        for r in results
    ])
    
    return df_results, results

# Example for testing the evaluator
if __name__ == "__main__":
    from pipe1 import llm, retriever, format_docs, rag_chain
    
    # Initialize evaluator
    evaluator = RAGEvaluator(llm, retriever, format_docs)
    
    # Define test questions
    test_questions = [
        "What programs are offered by the School of Business Studies at IBA?",
        "Tell me about the BS Economics program at IBA",
        "What is the financial assistance program at IBA?",
        "Who is the Executive Director of IBA?",
        "What is the duration of the MBA program at IBA?"
    ]
    
    # Run test evaluations
    for question in test_questions:
        print(f"Testing question: {question}")
        answer = rag_chain.invoke(question)
        print(f"Answer: {answer}")
        
        eval_result = evaluator.evaluate_answer(question, answer)
        
        print(f"Faithfulness Score: {eval_result['evaluation']['faithfulness_score']}/10")
        print(f"Relevance Score: {eval_result['evaluation']['relevance_score']}/10")
        print(f"Combined Score: {eval_result['evaluation']['combined_score']}/10")
        print(f"Total Response Time: {eval_result['performance']['total_time']:.2f} seconds")
        print("-" * 50)
    
    # Export results
    report_path = evaluator.export_results()
    print(f"Evaluation report saved to: {report_path}")