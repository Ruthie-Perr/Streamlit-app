import streamlit as st
import pandas as pd
import numpy as np
import openai
import pdfplumber
from openai import OpenAI
from io import BytesIO
import chromadb
from chromadb.config import Settings
from chromadb.utils.embedding_functions import OpenAIEmbeddingFunction

# -----------------------------
# Config
# -----------------------------
openai.api_key = st.secrets["OPENAI_API_KEY"]
client = OpenAI(api_key=openai.api_key)
EMBED_MODEL = "text-embedding-3-small"

st.set_page_config(page_title="AEM-Cube Result Area Generator", layout="wide")
st.title("📊 AEM-Cube Result Area Generator")
st.markdown("Upload a job profile PDF. The app will generate 4–5 result areas and AEM-Cube score bandwidths using RAG.")

# -----------------------------
# Step 1: Load and Embed Examples
# -----------------------------
@st.cache_resource(show_spinner=False)
def load_index():
    df = pd.read_csv("Resultaatgebieden Excel - Sheet1.csv")
    df.dropna(subset=["job_description", "output"], inplace=True)

    embed_fn = OpenAIEmbeddingFunction(api_key=openai.api_key, model_name=EMBED_MODEL)
    db = chromadb.Client(Settings(allow_reset=True))
    db.reset()

    collection = db.create_collection(name="aem_examples", embedding_function=embed_fn)

    documents = df["job_description"].tolist()
    metadatas = [{"output": out} for out in df["output"]]
    ids = [f"ex_{i}" for i in range(len(documents))]

    collection.add(documents=documents, metadatas=metadatas, ids=ids)
    return collection

collection = load_index()

# -----------------------------
# Step 2: Upload Job Profile PDF
# -----------------------------
uploaded_file = st.file_uploader("📄 Upload Job Profile PDF", type="pdf")

def extract_text_from_pdf(file):
    with pdfplumber.open(file) as pdf:
        return "\n".join(page.extract_text() or "" for page in pdf.pages)

if uploaded_file:
    with st.spinner("🔍 Extracting and embedding job profile..."):
        job_text = extract_text_from_pdf(uploaded_file)
        result = collection.query(query_texts=[job_text], n_results=1)
        matched_example = result["metadatas"][0][0]["output"]

    st.success("✅ Similar example retrieved. Generating result areas...")
    with st.expander("📌 Retrieved Example"):
        st.markdown(matched_example)

    # -----------------------------
    # Step 3: Build Prompt and Call GPT
    # -----------------------------
    prompt = f"""
You are an expert in job profile analysis using the AEM-Cube framework.

Here is an example output:
---
{matched_example}
---

Now analyze the following job profile:
---
{job_text}
---

Generate 4–5 result areas. For each, include:
- A title
- A short description
- Suggested AEM-Cube bandwidths (Attachment, Exploration, Managing Complexity)

Only return the output in the same format as the example.
"""

    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "You are a structured and precise job profile interpreter."},
            {"role": "user", "content": prompt}
        ],
        temperature=0.4
    )

    st.subheader("🧠 Generated Result Areas")
    st.markdown(response.choices[0].message.content)
