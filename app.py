import streamlit as st
from neo4j import GraphDatabase
from sentence_transformers import SentenceTransformer, util
import torch
import openai
import openai.error
import os
from dotenv import load_dotenv

load_dotenv()

# Set OpenAI API key
openai_api_key = os.getenv("OPENAI_API_KEY")
if not openai_api_key:
    st.error("OpenAI API key is not set. Please set the OPENAI_API_KEY environment variable.")
    raise ValueError("OpenAI API key is not set.")
openai.api_key = openai_api_key

# Neo4j connection details
NEO4J_URI = os.getenv("NEO4J_URI")  
NEO4J_USERNAME = os.getenv("NEO4J_USERNAME")              
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD")  

# Initialize Neo4j driver
driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USERNAME, NEO4J_PASSWORD))

# Load the NLP model
model = SentenceTransformer('all-MiniLM-L6-v2')

def compute_embedding(text):
    """Compute the embedding of the input text."""
    return model.encode(text, convert_to_tensor=True).float()

def find_most_similar_question(user_query, similarity_threshold=0.3):
    """Find the most similar question in Neo4j to the user's query."""
    user_embedding = compute_embedding(user_query)
    # Reshape user_embedding to [1, embedding_dim]
    user_embedding = user_embedding.unsqueeze(0)

    with driver.session() as session:
        # Fetch all questions and embeddings from Neo4j
        result = session.run(
            """
            MATCH (q:Question)-[:HAS_ANSWER]->(a:Answer)
            RETURN q.text AS question_text, q.embedding AS embedding, a.text AS answer_text
            """
        )

        questions = []
        embeddings = []
        answers = []

        for record in result:
            question_text = record['question_text']
            embedding = record['embedding']
            answer_text = record['answer_text']
            if embedding:
                questions.append(question_text)
                # Convert embedding to torch tensor with float32 dtype
                embeddings.append(torch.tensor(embedding, dtype=torch.float32))
                answers.append(answer_text)

    if not embeddings:
        return None

    # Stack embeddings into a tensor
    embeddings = torch.stack(embeddings)

    # Compute cosine similarities
    similarities = util.cos_sim(user_embedding, embeddings)[0]
    top_match_idx = torch.argmax(similarities).item()
    top_similarity = similarities[top_match_idx].item()

    if top_similarity >= similarity_threshold:
        best_question = questions[top_match_idx]
        best_answer = answers[top_match_idx]
        return {
            'question': best_question,
            'answer': best_answer,
            'similarity': top_similarity
        }

    # No similar question found above the threshold
    # Fallback to OpenAI's GPT-3.5-turbo
    try:
        assistant_response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
                    # Start of Selection
                    {"role": "system", "content": "You are a helpful and professional customer support agent for Thoughtful AI. Provide accurate and courteous responses to user inquiries while ensuring compliance with all company policies and guidelines. Avoid disallowed content or any inappropriate information. If the user's request violates policy, politely decline and encourage them to rephrase or ask a different question."},
                    {"role": "user", "content": user_query}
            ],
            timeout=15  # Optional timeout in seconds
        )
        response_text = assistant_response['choices'][0]['message']['content'].strip()
    except openai.error.OpenAIError as e:
        st.error(f"OpenAI API error: {e}")
        response_text = "I'm sorry, I'm unable to process your request at the moment."

    return {
        'question': None,
        'answer': response_text,
        'similarity': None
    }

# Initialize the database only once
if 'initialized' not in st.session_state:
    # No need to re-initialize the database; assume it's already populated
    st.session_state['initialized'] = True

# Streamlit app
st.title("Thoughtful AI Support Chatbot")

# Maintain conversation history
if 'history' not in st.session_state:
    st.session_state['history'] = []

# User input
user_input = st.text_input("You:", key='input')

if user_input:
    st.session_state['history'].append(("You", user_input))

    # Get the assistant's response
    result = find_most_similar_question(user_input)
    response = result['answer']

    st.session_state['history'].append(("Assistant", response))

# Display conversation history
for speaker, message in st.session_state['history']:
    if speaker == "You":
        st.markdown(f"**You:** {message}")
    else:
        st.markdown(f"**Assistant:** {message}")
