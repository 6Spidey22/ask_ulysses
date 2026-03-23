import ollama

from sentence_transformers import SentenceTransformer
import chromadb
from huggingface_hub import login

#Passing hugging face token to
hf_token = "" #put your own hugging face token here
login(token=hf_token)

#Load a pre-trained model to convert text into vectors
model = SentenceTransformer('all-MiniLM-L6-v2')

#Initialize Chroma DB client and create a collection
client = chromadb.PersistentClient(path="./vector_db") # Stores the DB locally
collection = client.get_or_create_collection(name="scraped_paragraphs")

question = "What is Cornell College block plan?"

results = collection.query(query_texts=[question], n_results=5 )

formatted_prompt = f"Question: {question}\n\nContext: {results['documents']}"

response = ollama.chat(model='llama3.1', messages=[
    {
        'role': 'user',
        'content': formatted_prompt,
    },
])
print(response['message']['content'])
