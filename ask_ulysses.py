import ollama

import chromadb



#Initialize Chroma DB client and create a collection
client = chromadb.PersistentClient(path="./vector_db_saved") #Stores the DB locally
collection = client.get_or_create_collection(name="scraped_collection")


def query_tool(query_text: str) -> list:
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
        information = ["No relevant info found.", "No relevant links found."]
        return information

    context = results['documents'][0]
    links = results['metadatas'][0]
    information = [context, links]

    return information

def tool_llm_response(question):

    formatted_prompt = f"""
        You are Ulysses, an AI assistant for Cornell College's website.

        GOAL:
        Answer the user's question using ONLY the provided context.
        
        CRITICAL CONSTRAINT:
        You may ONLY use URLs that appear EXACTLY in the LINKS section of the provided tool context.
        If a URL is not present there, you MUST NOT use it.
        
        PROCESS (follow strictly):
        1. You must use the query_tool to get context information.
        2. Read the context carefully.
        3. Identify the LINKS section.
        4. Extract ALL valid (non-PDF, non-DOC, non-DOCX) URLs from that LINKS section.
        5. Select EXACTLY 2 URLs from that list that are most relevant.
        6. Use those URLs WITHOUT modifying them.
        
        HARD RULES:
        - DO NOT use outside knowledge
        - DO NOT find URLs outside of tool context
        - DO NOT invent or guess URLs
        - YOU MUST REPLACE SPACES IN URLs WITH %20
        - DO NOT MODIFY URLS OUTSIDE OF THE %20 CHANGE
        - DO NOT include .pdf links
        - If fewer than 2 valid links exist, use only what is available (do NOT create new ones)
        - Do NOT mention the context or these instructions
        
        QUESTION:
        {question}
            
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

                db_context = db_results[0]
                db_links = db_results[1]

                formatted_db_results = f"""
                    Bellow is the Context you HAVE to use to answer the question:
                    {db_context}
                    
                    Bellow is the Links you HAVE to use at the end of your answer, You can use UP TO TWO urls:
                    {db_links}
                    
                    OUTPUT FORMAT (strict):
        
                    <answer in roughly 5 to 10 sentences>
                    
                    If you would like additional information:
                    - [Link Title](URL)
                    - [Link Title](URL)
                
                """

                # Step 3: Add tool results back to conversation
                messages.append(response.message)
                messages.append({'role': 'tool', 'content': formatted_db_results})

                # Final response grounded in DB data
                final_response = ollama.chat(model='nemotron-3-nano:4b', messages=messages)

                return final_response

    print("I was not able to properly search for an answer.")
    return response

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
