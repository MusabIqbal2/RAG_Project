from dotenv import load_dotenv
import os
from langchain_nomic import NomicEmbeddings
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_pinecone import PineconeVectorStore
from pinecone import Pinecone
from langchain_groq import ChatGroq
from langchain.prompts import PromptTemplate, ChatPromptTemplate, HumanMessagePromptTemplate



load_dotenv(override=True) # Load environment variables from .env file, override any existing variables

# Making a Langchain Embeddings Object using Nomic
embeddings = NomicEmbeddings(model="nomic-embed-text-v1.5")

# Making a Pinecone Vector Store Object
pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
index_name = "musab-bilal-rag"  
index = pc.Index(index_name)
vector_store = PineconeVectorStore(index=index, embedding=embeddings, namespace="prog-ann")

# Making a Retriever Object (Allows you to find similar documents in your Pinecone index, given a query)
retriever = vector_store.as_retriever(
    search_type="similarity_score_threshold",
    search_kwargs={"k": 10, "score_threshold": 0.5},
)

# Making a ChatGroq Object (This is the LLM model that will generate responses)
llm = ChatGroq(model="llama3-8b-8192", stop_sequences= None, temperature=0)

# Function to format the retrieved documents, gotten from the retriever

def format_docs(docs):
    print("docs:", docs)
    print()
    return "\n\n".join(doc.page_content for doc in docs)


# Making a custon prompt which had two variables, "context" and question

# Note:This prompt_template expects a dictionary/JSON with the keys "context" and "question" as input


prompt_template = ChatPromptTemplate.from_messages([
    HumanMessagePromptTemplate(
        prompt=PromptTemplate(
            input_variables=["context", "question"],
            template=( # The constructed prompt from the variables
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

# A simple function that logs the input and returns it

def logger(input):
    print(input)
    return input


# A chain with the modified prompt



# The chain simply looks likes this:
rag_chain = (
    {
        "context": retriever | format_docs,
        "question": RunnablePassthrough()
    }
    | prompt_template
    | llm
    | StrOutputParser()
)

# respnse = rag_chain.invoke("Tell me about the paper: Attention is all you Need")
print("dscs")