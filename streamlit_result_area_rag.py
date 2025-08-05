import streamlit as st
import pandas as pd
import numpy as np
import faiss
import openai
import pdfplumber
from openai import OpenAI
from io import BytesIO

# -----------------------------
# Config
# -----------------------------
openai.api_key = st.secrets["OPENAI_API_KEY"]
client = OpenAI(api_key=openai.api_key)
EMBED_MODEL = "text-embedding-3-small"
EMBED_DIM = 1536

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

    example_embeddings = []
    for jd in df["job_description"]:
        response = client.embeddings.create(input=jd, model=EMBED_MODEL)
        example_embeddings.append(response.data[0].embedding)

    index = faiss.IndexFlatL2(EMBED_DIM)
    index.add(np.array(example_embeddings).astype("float32"))
    return index, df["output"].tolist()

index, output_refs = load_index()

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
        response = client.embeddings.create(input=job_text, model=EMBED_MODEL)
        query_embed = np.array([response.data[0].embedding], dtype="float32")
        D, I = index.search(query_embed, k=1)
        matched_example = output_refs[I[0][0]]

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
