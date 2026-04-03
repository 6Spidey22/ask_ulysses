import ollama

import chromadb



#Initialize Chroma DB client and create a collection
client = chromadb.PersistentClient(path="./vector_db_saved") #Stores the DB locally
collection = client.get_or_create_collection(name="scraped_collection")


def query_tool(query_text: str) -> str:
    """
        Search the local vector database for information relevant to the user's query.
        Args:
            query_text: The search string to look up in the database. Don't include Cornell or College in this string
        Returns:
            The most relevant text snippet found, and find some links to use to give more information
    """

    embedded_prompt = ollama.embed(
        model='nomic-embed-text',
        input=query_text
    )

    query_embedding = embedded_prompt["embeddings"][0]

    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=20,
        include=['documents', 'metadatas']
    )

    if not results["documents"] or not results["documents"][0]:
        return "No relevant info found."

    context = "\n\n".join(results['documents'][0])
    context.join(f"\n\nLINKS section: {results['metadatas'][0]}")

    return context

def tool_llm_response(question):

    formatted_prompt = f"""
        You are Ulysses, an AI assistant for Cornell Colleges website to help students.

        GOAL:
        Provide a brief, helpful answer to the user’s question using ONLY the provided context.

        RULES:
        - DO NOT USE OUTSIDE KNOWLEDGE
        - Do NOT mention the context explicitly
        - Keep the response around 10 sentences at most
        - ALWAYS USE THE query_tool TOOL AND ONLY USE INFORMATION GIVEN BY THE TOOL
        
        
        LINK RULES:
        - Must include 2 links ONLY from provided tool context LINKS section, if you use any other links the answer is wrong
        - Do NOT include any links with .pdf
        - Must follow link format
        - DO NOT modify any urls
        
        
        *DO NOT include any Note sections saying you have followed specific instructions
        
        QUESTION:
        {question}
        
        OUTPUT FORMAT (strictly follow this):
        
        <your answer here>
        
        If you would like additional information:
        - [Link Title](URL)
        - [Link Title](URL)
            
    """

    messages =[{
        'role': 'user',
        'content':formatted_prompt,
    }]

    response = ollama.chat(
        model='nemotron-3-nano:4b',
        messages= messages,
        think = True,
        tools = [query_tool],
    )

    if response.message.tool_calls:
        for call in response.message.tool_calls:
            if call.function.name == 'query_tool':
                db_results = query_tool(**call.function.arguments)

                # Step 3: Add tool results back to conversation
                messages.append(response.message)
                messages.append({'role': 'tool', 'content': db_results})

                # Final response grounded in DB data
                final_response = ollama.chat(model='nemotron-3-nano:4b', messages=messages)

                return final_response

    return "I was not able to properly search for an answer."

def llm_response(question):
    embedded_prompt = ollama.embed(
        model='nomic-embed-text', #all-minilm',
        input=question
    )

    query_embedding = embedded_prompt["embeddings"][0]

    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=20,
        include=['documents', 'metadatas']
    )

    if not results["documents"] or not results["documents"][0]:
        print("No results found.")
        return {"message": {"content": "No relevant info found."}}

    context = "\n\n".join(results['documents'][0])
    links = results['metadatas'][0]

    formatted_prompt = f"""
        You are Ulysses, an AI assistant for Cornell Colleges website to help students.

        GOAL:
        Provide a brief, helpful answer to the user’s question using ONLY the provided context.

        RULES:
        - Do NOT use outside knowledge
        - Do NOT mention the context explicitly
        - Keep the response around five sentences, can be longer or shorter depending on if more or less info is needed.
        - Do NOT mention the context explicitly

        QUESTION:
        {question}

        CONTEXT:
        {context}

        LINKS:
        {links}

        OUTPUT FORMAT (strictly follow this):

        <your answer here>

        If you would like additional information:
        - [Link Title, clickable](URL)
        - [Link Title, clickable](URL)

        LINK RULES:
        - Use ONLY links from the LINKS section
        - Replace spaces in URLs with %20
        - Do NOT include any links with .pdf
        - Include 1–2 links
        - Must follow link format

        *DO NOT include any Note sections saying you have followed specific instructions

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
        talk = tool_llm_response(receive)
        print(talk['message']['content'])
