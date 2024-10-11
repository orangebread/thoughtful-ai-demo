# Thoughtful AI Customer Support Agent

This project implements a customer support AI agent for Thoughtful AI using a Retrieval-Augmented Generation (RAG) strategy. It leverages Neo4j for storing and retrieving question-answer pairs, and uses SentenceTransformers for generating embeddings.

## Demo

![Demo](demo.gif)

## Features

- **Retrieval-Augmented Generation (RAG)**: Combines information retrieval with generative AI to provide accurate and contextually relevant responses.
- **Streamlit Interface**: A user-friendly web application built with Streamlit for interactive conversations.
- **Predefined Knowledge Base**: Contains a set of predefined questions and answers about Thoughtful AI's agents stored in Neo4j.
- **Customizable Threshold**: Adjust the similarity match threshold to fine-tune the retrieval sensitivity.
- **Neo4j Integration**: Efficiently stores and retrieves question-answer pairs and their embeddings.

## Architecture Overview

This project implements a Retrieval-Augmented Generation (RAG) system using the following components:

1. **Neo4j Database**: Stores question-answer pairs and their corresponding embeddings.
2. **SentenceTransformer**: Generates embeddings for questions, enabling semantic similarity search.
3. **Streamlit Interface**: Provides a user-friendly web interface for interacting with the chatbot.

The system works as follows:

1. Questions and answers are pre-populated in the Neo4j database using the `generate_embeddings.py` script.
2. When a user asks a question, the system computes its embedding and performs a similarity search in Neo4j.
3. If a similar question is found above the similarity threshold, its corresponding answer is returned.
4. If no similar question is found, the system suggests related topics based on the top similar questions.

This architecture allows for efficient retrieval of relevant information without the need for real-time API calls to large language models, improving response time and reducing costs.

## Prerequisites

Before running the application, ensure you have the following:

1. **Python 3.7 or Higher**
   - Verify your Python version: `python --version`

2. **Neo4j Database**
   - Install and set up a Neo4j database instance.
   - Note down the URI, username, and password for configuration.

## Setup

1. **Clone the Repository**
   ```bash
   git clone https://github.com/orangebread/thoughtful-ai-demo.git
   cd thoughtful-ai-demo
   ```

2. **Create a Virtual Environment** (Optional but Recommended)
   ```bash
   python -m venv .venv
   # Activate the virtual environment:
   # On Windows:
   .venv\Scripts\Activate.ps1
   # On macOS/Linux:
   source .venv/bin/activate
   ```

3. **Install Dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Set Up Environment Variables**
   Create a `.env` file in the root directory with the following content:
   ```dotenv
   NEO4J_URI=bolt://localhost:7687
   NEO4J_USERNAME=your-neo4j-username
   NEO4J_PASSWORD=your-neo4j-password
   ```
   Replace the values with your actual Neo4j connection details.

5. **Populate the Neo4j Database**
   Run the embedding generation script:
   ```bash
   python generate_embeddings.py
   ```

6. **Run the Application**
   Start the Streamlit app:
   ```bash
   streamlit run app.py
   ```
   The application will open in your default web browser or navigate to `http://localhost:8501`.

## Using the Chat Interface

1. **Interact with the Agent**: Type questions about Thoughtful AI's agents in the chat interface.
2. **Receive Responses**: The AI agent retrieves relevant information and generates a response.
3. **Continue the Conversation**: Ask follow-up questions or new queries as needed.

## Customization

To expand or modify the agent's knowledge base:

1. **Add More Questions and Answers**
   - Edit the `qa_list` in `generate_embeddings.py`.
   - Add new question-answer pairs as dictionaries.

2. **Regenerate Embeddings and Update Neo4j Database**
   - Run `python generate_embeddings.py` after making changes.
   - This will recompute embeddings and update the Neo4j database.

3. **Restart the Application**
   - Rerun `streamlit run app.py` to load the updated data.

4. **Verify the Update**
   - Test new questions in the chat interface.
   - Optionally, check the Neo4j database directly using Cypher queries.

## Error Handling

- Catches and logs unexpected errors during execution.
- Provides user-friendly error messages in the Streamlit interface.

## Performance Optimization

- Uses SentenceTransformer for efficient embedding generation.
- Leverages Neo4j for fast similarity searches.
- Loads models and establishes database connections once at startup to reduce latency.

## Neo4j Database Management

- The database stores Question nodes linked to Answer nodes.
- Each Question node contains the question text and its embedding.
- Basic operations can be performed using Cypher queries through the Neo4j Browser.

## Security

- Store sensitive information (Neo4j credentials) using environment variables.
- Regularly update dependencies to incorporate security patches.

## Dependencies

All required Python packages are listed in `requirements.txt`, including:
- streamlit
- neo4j
- sentence-transformers
- python-dotenv
- torch

Ensure all dependencies are installed before running the application.