# %%
print("fdvaf")

# %%
from huggingface_hub import InferenceClient
import os
from dotenv import load_dotenv

load_dotenv()
client = InferenceClient(
	# provider="together", # optional, default is huggingface's own inference API
	api_key = os.getenv("HUGGINGFACE_API_KEY")
)

messages = [
	{
		"role": "user",
		"content": "What is the capital of France?"
	}
]

completion = client.chat.completions.create(
    model="meta-llama/Llama-2-7b-chat", 
	messages=messages, 
	max_tokens=500
)

print(completion.choices[0].message.content)


# %%
curl https://api.groq.com/openai/v1/chat/completions -s \
-H "Content-Type: application/json" \
-H "Authorization: Bearer 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" \
-d '{
"model": "meta-llama/llama-4-scout-17b-16e-instruct",
"messages": [{
    "role": "user",
    "content": "Explain the importance of fast language models"
}] 
}'

# %%
curl https://api.groq.com/openai/v1/chat/completions -s \
-H "Content-Type: application/json" \
-H "Authorization: Bearer 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" \
-d '{ 
"model": "meta-llama/llama-4-scout-17b-16e-instruct",
"messages": [{
    "role": "user",
    "content": "Explain the importance of fast language models"
}]
}'

# %%
from pinecone.grpc import PineconeGRPC as Pinecone
from dotenv import load_dotenv
import os
load_dotenv()


pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
print("Pinecone Indices: ", pc.list_indexes())

index = pc.Index(name="INDEX_NAME")

upsert_response = index.upsert(
    vectors=[
        {
            "id": "vec1", # unique string identifier for the vector, must be provided
            "values": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8], # put the embedding vector here
            "metadata": {  # put the actual document's text here
                "text": "This is a sample document.",
                "genre": "documentary" # other optional metadata
            }
        },
    ],
    namespace="example-namespace" # optional, defaults to "default"
)

# Finding similar vectors
index.query(
    namespace="example-namespace",
    vector=[0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3], # put the query vector here
    filter={ # optional, to filter the results based on metadata
        "genre": {"$eq": "documentary"}
    },
    top_k=3,
    include_values=True # optional, to include the vector values in the response
)
# %%
import os
from dotenv import load_dotenv
from groq import Groq

load_dotenv()

client = Groq(api_key=os.environ.get("GROQ_API_KEY"))

chat_completion = client.chat.completions.create(
    messages=[
        {
            "role": "system",
            "content": "you are a helpful assistant for question answering"
        },
        {
            "role": "user",
            "content": "Hi, how are you?",
        }
    ],
    model="llama-3.3-70b-versatile",
)

print(chat_completion.choices[0].message.content)
# %%

from nomic import embed
import numpy as np

output = embed.text(
    texts=['The text you want to embed.'],
    model='nomic-embed-text-v1.5',
    task_type='search_document',
)

embeddings = np.array(output['embeddings'])
print(embeddings[0].shape)  # prints: (768,)

# %%
from tavily import TavilyClient
import os
from dotenv import load_dotenv

load_dotenv()

# Step 1. Instantiating your TavilyClient
travily_client = TavilyClient(api_key=os.getenv("TAVILY_API_KEY"))

# Step 2. Executing the search request
response = travily_client.search("Who is Leo Messi?", max_results=10)

# Step 3. Printing the search results
for result in response["results"]:
    print(result["url"])
# %%
from dotenv import load_dotenv
import os

load_dotenv(override=True) 

from langchain_groq import ChatGroq
from langchain_nomic import NomicEmbeddings

groq = ChatGroq(model="llama3-8b-8192", stop_sequences=None, temperature=0)
response = groq.invoke("tell me all about Facebook")
print(response.content)
# %%
from langchain_openai import ChatOpenAI

llm = ChatOpenAI()
llm.invoke("Hello, world!")
# %%
from langchain_community.embeddings import NomicEmbeddings

# %%
