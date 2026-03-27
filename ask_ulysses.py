import ollama

import chromadb



#Initialize Chroma DB client and create a collection
client = chromadb.PersistentClient(path="./vector_db") #Stores the DB locally
collection = client.get_or_create_collection(name="scraped_collection")

def llm_response(question):
    embedded_prompt = ollama.embed(
        model='nomic-embed-text', #all-minilm',
        input=question
    )

    query_embedding = embedded_prompt["embeddings"][0]

    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=10,
        include=['documents']
    )

    if not results["documents"] or not results["documents"][0]:
        print("No results found.")
        return {"message": {"content": "No relevant info found."}}

    context = "\n\n".join(results['documents'][0])


    formatted_prompt = f"""
    Answer the question using ONLY the context below.
    
    Context:
    {context}
    
    Question: 
    {question}
    """

    response = ollama.chat(model='llama3.1', messages=[
        {
            'role': 'user',
            'content': formatted_prompt,
        },
    ])
    print(f"\n\n{results['documents'][0]}\n\n")

    return response


if __name__ == '__main__':
    print(collection.count())
    while True:
        receive = input("Question: ")#"What is Cornell College block plan?"
        if receive == "quit":
            break
        talk = llm_response(receive)
        print(talk['message']['content'])
