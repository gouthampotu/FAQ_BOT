import streamlit as st
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# ------------------------------
# Page Configuration
# ------------------------------

st.set_page_config(
    page_title="FAQ Intelligence Bot",
    page_icon="🤖",
    layout="centered"
)

st.title("🤖 FAQ Intelligence Bot")
st.write("Ask any question from the FAQ knowledge base.")

# ------------------------------
# FAQ Dataset
# ------------------------------

faq_data = {
    "Question": [
        "What is Artificial Intelligence?",
        "What is Machine Learning?",
        "What is Deep Learning?",
        "What is Python?",
        "What is Data Science?",
        "What is Generative AI?",
        "What is ChatGPT?",
        "What is NLP?"
    ],

    "Answer": [
        "Artificial Intelligence is the simulation of human intelligence by machines.",

        "Machine Learning is a subset of AI that enables systems to learn from data.",

        "Deep Learning is a subset of Machine Learning that uses neural networks.",

        "Python is a popular programming language widely used in AI and Data Science.",

        "Data Science is the process of extracting useful insights from data.",

        "Generative AI creates new content such as text, images, audio, and code.",

        "ChatGPT is a conversational AI model developed by OpenAI.",

        "NLP stands for Natural Language Processing, which helps computers understand human language."
    ]
}

df = pd.DataFrame(faq_data)

# ------------------------------
# Load Embedding Model
# ------------------------------

@st.cache_resource
def load_model():
    return SentenceTransformer("all-MiniLM-L6-v2")

model = load_model()

# ------------------------------
# Generate FAQ Embeddings
# ------------------------------

@st.cache_data
def create_embeddings():
    return model.encode(df["Question"].tolist())

faq_embeddings = create_embeddings()

# ------------------------------
# Semantic Search Function
# ------------------------------

def get_answer(user_query):

    query_embedding = model.encode([user_query])

    similarities = cosine_similarity(
        query_embedding,
        faq_embeddings
    )[0]

    best_match_index = similarities.argmax()

    answer = df.iloc[best_match_index]["Answer"]

    matched_question = df.iloc[best_match_index]["Question"]

    confidence = similarities[best_match_index]

    return answer, matched_question, confidence

# ------------------------------
# User Interface
# ------------------------------

query = st.text_input(
    "Ask your question:",
    placeholder="Example: Explain AI"
)

if st.button("Get Answer"):

    if query.strip() == "":
        st.warning("Please enter a question.")

    else:

        answer, matched_question, confidence = get_answer(query)

        st.success("Answer Found")

        st.subheader("Answer")
        st.write(answer)

        st.subheader("Matched FAQ")
        st.write(matched_question)

        st.subheader("Similarity Score")
        st.write(round(float(confidence), 3))

# ------------------------------
# Footer
# ------------------------------

st.markdown("---")
st.caption("Built using Streamlit + Sentence Transformers + Cosine Similarity")