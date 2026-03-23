import ollama

from sentence_transformers import SentenceTransformer
from huggingface_hub import login
import chromadb

#Passing hugging face token to
hf_token = "" #put your hugging face token here
login(token=hf_token)

#Load a pre-trained model to convert text into vectors
model = SentenceTransformer('all-MiniLM-L6-v2')

#Initialize Chroma DB client and create a collection
client = chromadb.PersistentClient(path="./vector_db") # Stores the DB locally
collection = client.get_or_create_collection(name="scraped_paragraphs")

question = input("Question: ")#"What is Cornell College block plan?"

results = collection.query(query_texts=[question], n_results=5 )

formatted_prompt = f"Question: {question}\n\nContext: {results['documents']}"

response = ollama.chat(model='llama3.1', messages=[
    {
        'role': 'user',
        'content': formatted_prompt,
    },
])
print(response['message']['content'])
