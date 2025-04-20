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
    "Tell me about the BS Economics program at IBA",
    "What is the financial assistance program at IBA?",
    "Who is the Executive Director of IBA?",
    "What is the duration of the MBA program at IBA?",
    "What is the full form of IBA?"
]

expected_responses = [
    """The School of Business Studies at IBA offers the following programs:
BBA (Bachelor of Business Administration)
BS (Accounting and Finance)
Bs Conputer Science
BS Economics
Bs Mathematics
BS Social Sciences
BS Economic and Mathematics
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
"""
The IBA Financial Assistance Program is designed to support meritorious students from all socio-economic backgrounds across Pakistan. Between 2008 and 2024, PKR 3.17 billion was granted in financial aid. In the academic year 2023–2024 alone, approximately 600 students benefited from financial assistance, and around 170 students received support through Qarz-e-Hasna.
The program offers multiple forms of assistance, including fee installments, Qarz-e-Hasna, and bridge financing (temporary financial relief while awaiting external donor scholarships such as Zakat or government/private sector funding). Eligible students are full-time undergraduates and graduates enrolled in morning programs (excluding EMBA).
The aid can cover 25% to 100% of tuition fees and is based on a need assessment conducted by the Financial Assistance Committee (FAC). Applicants may be required to attend interviews, submit supporting documents, or undergo home verifications. Students must maintain a minimum CGPA of 2.5 to continue receiving support and must reapply annually.
Additionally, financial assistance is only provided for active semester tuition and does not cover arrears, repeated, or withdrawn courses. Misrepresentation or failure to comply with the guidelines can result in revocation of aid and disciplinary action.
For further inquiries, students are advised to contact the Financial Assistance Office at financial-aid@iba.edu.pk or visit https://www.iba.edu.pk/installments to apply.
""",
"The executive director of IBA is Dr. S. Akbar Zaidi.",
"The Duration of the MBA program at IBA is 2 years.",
"IBA stands for Institute of Business Administration"

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
    metrics=[LLMContextRecall(), Faithfulness(), FactualCorrectness(),ContextEntityRecall(),NoiseSensitivity()],
    llm=evaluator_llm,
    run_config=RunConfig(timeout=1200) 
)
print("result")
print(result)