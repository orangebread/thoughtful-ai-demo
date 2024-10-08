from neo4j import GraphDatabase
from sentence_transformers import SentenceTransformer
import numpy as np

# Neo4j connection details
from dotenv import load_dotenv
import os

load_dotenv()

NEO4J_URI = os.getenv('NEO4J_URI')
NEO4J_USERNAME = os.getenv('NEO4J_USERNAME')
NEO4J_PASSWORD = os.getenv('NEO4J_PASSWORD')

# Initialize the Neo4j driver
driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USERNAME, NEO4J_PASSWORD))

# Load the Sentence Transformer model
model = SentenceTransformer('all-MiniLM-L6-v2')

# Predefined dataset of questions and answers
qa_list = [
    {
        "question": "What does the eligibility verification agent (EVA) do?",
        "answer": "EVA automates the process of verifying a patient’s eligibility and benefits information in real-time, eliminating manual data entry errors and reducing claim rejections."
    },
    {
        "question": "What does the claims processing agent (CAM) do?",
        "answer": "CAM streamlines the submission and management of claims, improving accuracy, reducing manual intervention, and accelerating reimbursements."
    },
    {
        "question": "How does the payment posting agent (PHIL) work?",
        "answer": "PHIL automates the posting of payments to patient accounts, ensuring fast, accurate reconciliation of payments and reducing administrative burden."
    },
    {
        "question": "Tell me about Thoughtful AI's Agents.",
        "answer": "Thoughtful AI provides a suite of AI-powered automation agents designed to streamline healthcare processes. These include Eligibility Verification (EVA), Claims Processing (CAM), and Payment Posting (PHIL), among others."
    },
    {
        "question": "What are the benefits of using Thoughtful AI's agents?",
        "answer": "Using Thoughtful AI's Agents can significantly reduce administrative costs, improve operational efficiency, and reduce errors in critical processes like claims management and payment posting."
    }
]

def compute_embeddings(questions):
    embeddings = {}
    for question in questions:
        embedding = model.encode(question)
        embeddings[question] = embedding.tolist()  # Convert numpy array to list
    return embeddings

def store_questions_with_embeddings(qa_list, embeddings):
    with driver.session() as session:
        # Optional: Clear existing data
        session.run("MATCH (n) DETACH DELETE n")

        for qa in qa_list:
            question_text = qa['question']
            answer_text = qa['answer']
            embedding = embeddings[question_text]

            # Create Question and Answer nodes with embedding
            session.run(
                """
                CREATE (q:Question {text: $question_text, embedding: $embedding})
                CREATE (a:Answer {text: $answer_text})
                CREATE (q)-[:HAS_ANSWER]->(a)
                """,
                question_text=question_text,
                answer_text=answer_text,
                embedding=embedding
            )
            print(f"Stored question: {question_text}")

# Extract questions from the dataset
questions = [qa['question'] for qa in qa_list]

# Step 3.1: Compute embeddings for the questions
embeddings = compute_embeddings(questions)

# Step 3.2: Store questions, embeddings, and answers in Neo4j
store_questions_with_embeddings(qa_list, embeddings)

# Close the Neo4j driver when done
driver.close()
