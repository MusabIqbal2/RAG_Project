from logging import config
from ragas.dataset_schema import SingleTurnSample
from ragas.metrics import (
    LLMContextRecall,
    Faithfulness,
    FactualCorrectness,
    LLMContextPrecisionWithoutReference,
    ContextEntityRecall,
    NoiseSensitivity,
)
from ragas import evaluate, EvaluationDataset, RunConfig
from ragas.llms import LangchainLLMWrapper
from pipe1 import llm, rag_chain, retriever,format_docs


sample_queries = [
    "What programs are offered by the School of Business Studies at IBA?",
    # "Tell me about the BS Economics program at IBA",
    # "What is the financial assistance program at IBA?",
    # "Who is the Executive Director of IBA?",
    # "What is the duration of the MBA program at IBA?",
]

expected_responses = [
    """The School of Business Studies at IBA offers the following programs:
BBA (Bachelor of Business Administration)
BS (Accounting and Finance)
MBA (Master of Business Administration)
EMBA (Executive Master of Business Administration)
MS (Master of Science) in various specializations, including:
Finance
Islamic Banking and Finance
Management
Marketing
PhD (Doctor of Philosophy) in various specializations, including:
Computer Science
Economics
Mathematics""",

]

evaluator_llm = LangchainLLMWrapper(llm)

dataset = []

for query, reference in zip(sample_queries, expected_responses):
    relevant_docs = retriever.invoke(query)
    response = rag_chain.invoke(query)
    dataset.append(
        {
            "user_input": query,
            "retrieved_contexts": [rdoc.page_content for rdoc in relevant_docs],
            "response": response,
            "reference": reference,
        }
    )

evaluation_dataset = EvaluationDataset.from_list(dataset)


result = evaluate(
    dataset=evaluation_dataset,
    metrics=[LLMContextRecall(), Faithfulness(), FactualCorrectness(),LLMContextPrecisionWithoutReference(),ContextEntityRecall(),NoiseSensitivity()],
    llm=evaluator_llm,
    run_config=RunConfig(timeout=1200) 
)
print("result")
print(result)