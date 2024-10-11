import streamlit as st
from neo4j import GraphDatabase
from sentence_transformers import SentenceTransformer, util
import torch
import os
from dotenv import load_dotenv
import logging
from typing import Dict, List, Tuple

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

# Neo4j connection details
NEO4J_URI = os.getenv("NEO4J_URI")
NEO4J_USERNAME = os.getenv("NEO4J_USERNAME")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD")

# Constants
MODEL_NAME = 'all-MiniLM-L6-v2'
SIMILARITY_THRESHOLD = 0.4
SUGGESTION_COUNT = 3

class Neo4jConnection:
    def __init__(self, uri: str, user: str, password: str):
        self._driver = GraphDatabase.driver(uri, auth=(user, password))

    def close(self):
        self._driver.close()

    def query(self, query: str, parameters: Dict = None) -> List[Dict]:
        with self._driver.session() as session:
            result = session.run(query, parameters)
            return [record.data() for record in result]

class EmbeddingModel:
    def __init__(self, model_name: str):
        self.model = SentenceTransformer(model_name)

    def compute_embedding(self, text: str) -> torch.Tensor:
        return self.model.encode(text, convert_to_tensor=True).float()

class SimilaritySearch:
    def __init__(self, neo4j_connection: Neo4jConnection, embedding_model: EmbeddingModel):
        self.neo4j_connection = neo4j_connection
        self.embedding_model = embedding_model

    def find_most_similar_question(self, user_query: str, similarity_threshold: float, suggestion_count: int) -> Dict:
        user_embedding = self.embedding_model.compute_embedding(user_query).unsqueeze(0)

        result = self.neo4j_connection.query(
            """
            MATCH (q:Question)-[:HAS_ANSWER]->(a:Answer)
            RETURN q.text AS question_text, q.embedding AS embedding, a.text AS answer_text
            """
        )

        questions, embeddings, answers = [], [], []
        for record in result:
            if record['embedding']:
                questions.append(record['question_text'])
                embeddings.append(torch.tensor(record['embedding'], dtype=torch.float32))
                answers.append(record['answer_text'])

        if not embeddings:
            logger.warning("No embeddings found in the database.")
            return None

        embeddings = torch.stack(embeddings)
        similarities = util.cos_sim(user_embedding, embeddings)[0]
        top_similarities, top_indices = torch.topk(similarities, k=suggestion_count)
        
        top_match_idx = top_indices[0].item()
        top_similarity = top_similarities[0].item()

        if top_similarity >= similarity_threshold:
            return {
                'question': questions[top_match_idx],
                'answer': answers[top_match_idx],
                'similarity': top_similarity,
                'suggestions': None
            }
        else:
            return {
                'question': None,
                'answer': None,
                'similarity': None,
                'suggestions': [questions[idx.item()] for idx in top_indices]
            }

class ChatbotUI:
    def __init__(self, similarity_search: SimilaritySearch):
        self.similarity_search = similarity_search

    def initialize_session_state(self):
        if 'history' not in st.session_state:
            st.session_state['history'] = []
        if 'suggestions' not in st.session_state:
            st.session_state['suggestions'] = []
        if 'last_user_input' not in st.session_state:
            st.session_state['last_user_input'] = ''
        if 'input_key' not in st.session_state:
            st.session_state['input_key'] = 0

    def render(self):
        st.title("Thoughtful AI Support Chatbot")
        self.initialize_session_state()

        # Chat history in a container
        chat_container = st.container()

        # Suggestions
        suggestion_container = st.container()
        
        # Custom CSS for send button alignment and styling
        st.markdown("""
        <style>
        .stButton > button[kind="primary"] {
            height: 3em;
            width: 5em;
            margin-top: 1.7em;
        }
        div.row-widget.stButton {
            text-align: right;
        }
        </style>
        """, unsafe_allow_html=True)

        # User input
        col1, col2 = st.columns([5, 1])
        with col1:
            user_input = st.empty()
            current_input = user_input.text_input("You:", key=f"input_{st.session_state['input_key']}")
        with col2:
            send_button = st.button("Send", key="send_button", type="primary")

        # Process user input
        if (current_input and current_input != st.session_state['last_user_input']) or send_button:
            if current_input:  # Ensure there's actually input to process
                st.session_state['last_user_input'] = current_input
                self.process_user_input(current_input)
                st.session_state['input_key'] += 1  # Increment key to clear input
                st.rerun()

        # Display chat history
        with chat_container:
            self.display_conversation_history()

        # Handle suggestion clicks
        with suggestion_container:
            self.handle_suggestion_clicks()

    def process_user_input(self, user_input: str):
        st.session_state['history'].append(("You", user_input))
        response, suggestions = self.get_response(user_input)
        st.session_state['history'].append(("Assistant", response))
        st.session_state['suggestions'] = suggestions

    def get_response(self, user_input: str) -> Tuple[str, List[str]]:
        result = self.similarity_search.find_most_similar_question(
            user_input, SIMILARITY_THRESHOLD, SUGGESTION_COUNT
        )

        if result is None:
            return "I'm sorry, I couldn't find an answer to your question.", []
        elif result['answer'] is not None:
            return result['answer'], []
        else:
            response = (
                "I'm sorry, I couldn't find an exact answer to your question. "
                "Here are some topics you might be interested in:"
            )
            return response, result['suggestions']

    def display_conversation_history(self):
        for speaker, message in st.session_state['history']:
            st.markdown(f"**{speaker}:** {message}")

    def handle_suggestion_clicks(self):
        if st.session_state['suggestions']:
            st.markdown("**Suggested topics:**")
            for suggestion in st.session_state['suggestions']:
                if st.button(suggestion, key=suggestion):
                    self.process_user_input(suggestion)
                    st.session_state['suggestions'] = []  # Clear suggestions after click
                    st.session_state['input_key'] += 1  # Increment key to clear input
                    st.rerun()

def main():
    try:
        neo4j_connection = Neo4jConnection(NEO4J_URI, NEO4J_USERNAME, NEO4J_PASSWORD)
        embedding_model = EmbeddingModel(MODEL_NAME)
        similarity_search = SimilaritySearch(neo4j_connection, embedding_model)
        chatbot_ui = ChatbotUI(similarity_search)

        chatbot_ui.render()

    except Exception as e:
        logger.error(f"An error occurred: {str(e)}")
        st.error("An unexpected error occurred. Please try again later.")

if __name__ == "__main__":
    main()