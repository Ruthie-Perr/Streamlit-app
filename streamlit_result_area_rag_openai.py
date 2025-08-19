import streamlit as st
import pandas as pd
import numpy as np
import pdfplumber
from io import BytesIO
from openai import OpenAI

# --- Config ---
client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])

# --- Embed text using OpenAI Embedding API ---
@st.cache_data(show_spinner=False)
def embed_text(texts):
    response = client.embeddings.create(
        model="text-embedding-3-small",
        input=texts
    )
    return np.array([r.embedding for r in response.data])

# --- Extract all text from a PDF ---
def extract_text_from_pdf(file):
    with pdfplumber.open(file) as pdf:
        return "\n".join([page.extract_text() or "" for page in pdf.pages])

# --- Compute cosine similarity ---
def cosine_similarity(a, b):
    a_norm = a / np.linalg.norm(a, axis=1, keepdims=True)
    b_norm = b / np.linalg.norm(b)
    return np.dot(a_norm, b)

# --- Load CSV and embed examples ---
@st.cache_data(show_spinner=False)
def load_examples():
    df = pd.read_csv("Resultaatgebieden Excel - Sheet1.csv")

    df["combined"] = (
        df["Resultaatgebied"] + "\n"
        + "Attachment: " + df["Attachment"].astype(str) + "\n"
        + "Exploratie: " + df["Exploratie"].astype(str) + "\n"
        + "Managen Complexiteit: " + df["Managen van Complexiteit"].astype(str)
    )

    embeddings = embed_text(df["combined"].tolist())
    return df, embeddings

# --- UI ---
st.title("📄 Result Area Generator")
uploaded_pdf = st.file_uploader("Upload Job Description (PDF)", type="pdf")

if uploaded_pdf:
    st.info("Reading job description and searching examples...")
    query_text = extract_text_from_pdf(uploaded_pdf)
    query_embed = embed_text([query_text])[0]

    df, example_embeds = load_examples()
    sims = cosine_similarity(example_embeds, query_embed)
    top_idx = sims.argsort()[::-1][:4]

    st.success("✅ Found relevant result areas:")
    for i in top_idx:
        st.markdown(f"**• {df.iloc[i]['Resultaatgebied']}**")
        st.markdown(f"Attachment: {df.iloc[i]['Attachment']}")
        st.markdown(f"Exploratie: {df.iloc[i]['Exploratie']}")
        st.markdown(f"Managen Complexiteit: {df.iloc[i]['Managen van Complexiteit']}")
        st.markdown("---")

    # Combine top matches for prompt context
    context = "\n".join(df.iloc[top_idx]["combined"].tolist())

    # Prompt GPT-3.5
    prompt = (
        "You are an expert in translating job descriptions into strategic Result Areas and AEM-Cube bandwidths.\n\n"
        f"Based on the following relevant examples:\n{context}\n\n"
        f"Now read this job description:\n{query_text}\n\n"
        "Generate 4–5 distinct Result Areas, each with a matching AEM Bandwidth range. "
        "Do not copy from the examples — infer what fits best. Format the output clearly and concisely.\n\n"
        "Respond in this format:\n\n"
        "1. [Result Area Title]\n"
        "AEM Bandwidth:\nAttachment: [range], Exploratie: [range], Managen Complexiteit: [range]\n"
        "[Optional 1-line explanation]\n\n"
        "Repeat for each."
    )

    with st.spinner("Generating result areas..."):
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are a highly precise job design expert."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.4
        )
        result = response.choices[0].message.content
        st.subheader("🔎 Suggested Result Areas")
        st.markdown(result)

