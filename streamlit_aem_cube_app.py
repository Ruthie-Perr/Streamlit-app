import streamlit as st
import pandas as pd
import numpy as np
import re
import pdfplumber
from io import BytesIO
from openai import OpenAI
import tiktoken

# -----------------------------
# Configuration
# -----------------------------
client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])
encoding = tiktoken.encoding_for_model("gpt-4")

# -----------------------------
# Load theory and examples
# -----------------------------
with open("Foundational Info (1).txt", "r", encoding="utf-8") as f:
    foundational_text = f.read()

with open("Descriptions (3).txt", "r", encoding="utf-8") as f:
    description_text = f.read()

def extract_theory_block(name, text):
    focus_areas = {
        "Product-Market Fit": ("1. Product-Market Fit", "2. Speed-to-Market"),
        "Speed-to-Market": ("2. Speed-to-Market", "3. Strategic Agility Index Score"),
        "Strategic Agility Index": ("3. Strategic Agility Index Score", "4. Strategic Hire Analysis"),
        "Strategic Hire Analysis": ("4. Strategic Hire Analysis", "5. Business Performance"),
        "Business Performance": ("5. Business Performance", "6. Safeguarding Innovation"),
        "Safeguarding Innovation": ("6. Safeguarding Innovation", "7. Summary"),
        "Summary": ("7. Summary", None)
    }
    start_tag, end_tag = focus_areas[name]
    section = text.split(start_tag)[1]
    return section.split(end_tag)[0].strip() if end_tag else section.strip()

def extract_example_block(name, text):
    if name in text:
        section = text.split(name)[1]
        return section.split("\n\n")[0].strip()
    return ""

def extract_scores_from_pdf(pdf_file):
    scores = []
    with pdfplumber.open(pdf_file) as pdf:
        for page in pdf.pages:
            text = page.extract_text()
            if not text:
                continue
            lines = text.splitlines()
            a = e = c = None
            for line in lines:
                line = line.strip().lower()
                if line.startswith("attachment"):
                    a_match = re.findall(r"\d+", line)
                    if a_match:
                        a = int(a_match[-1])
                elif line.startswith("exploratie") or line.startswith("exploration"):
                    e_match = re.findall(r"\d+", line)
                    if e_match:
                        e = int(e_match[-1])
                elif line.startswith("managen") or line.startswith("managing"):
                    c_match = re.findall(r"\d+", line)
                    if c_match:
                        c = int(c_match[-1])
            if a is not None and e is not None and c is not None:
                scores.append({
                    "Project": "Team zelfbeelden",
                    "attachment score": a,
                    "exploration score": e,
                    "managing complexity score": c
                })
    return pd.DataFrame(scores)

def generate_team_score_summary(focus, test_team):
    aggregate_measures = {}
    for dimension in ["attachment score", "exploration score", "managing complexity score"]:
        scores = test_team[dimension]
        aggregate_measures[dimension] = {
            "mean": np.round(scores.mean(), 2),
            "std_dev": np.round(scores.std(), 2),
            "ratio_below_25": np.round((scores < 25).mean(), 2),
            "ratio_25_50": np.round(((scores >= 25) & (scores < 50)).mean(), 2),
            "ratio_50_75": np.round(((scores >= 50) & (scores < 75)).mean(), 2),
            "ratio_above_75": np.round((scores >= 75).mean(), 2)
        }

    ratio_high_high = np.round(((test_team["attachment score"] > 65) & (test_team["exploration score"] > 65)).mean(), 2)
    ratio_mid_mid = np.round(((test_team["attachment score"].between(35, 65)) & (test_team["exploration score"].between(35, 65))).mean(), 2)
    ratio_low_low = np.round(((test_team["attachment score"] < 35) & (test_team["exploration score"] < 35)).mean(), 2)
    ratio_low_high = np.round(((test_team["attachment score"] < 35) & (test_team["exploration score"] > 65)).mean(), 2)
    ratio_high_low = np.round(((test_team["attachment score"] > 65) & (test_team["exploration score"] < 35)).mean(), 2)
    ratio_quadrant_1 = np.round(((test_team["attachment score"] < 51) & (test_team["exploration score"] < 51)).mean(), 2)
    ratio_quadrant_2 = np.round(((test_team["attachment score"] > 50) & (test_team["exploration score"] < 51)).mean(), 2)
    ratio_quadrant_3 = np.round(((test_team["attachment score"] > 50) & (test_team["exploration score"] > 50)).mean(), 2)
    ratio_quadrant_4 = np.round(((test_team["attachment score"] < 51) & (test_team["exploration score"] > 50)).mean(), 2)
    ratio_quadrant_5 = np.round(((test_team["attachment score"].between(37.5, 62.5)) & (test_team["exploration score"].between(37.5, 62.5))).mean(), 2)

    focus_to_dimensions = {
        "Product-Market Fit": ["attachment score"],
        "Speed-to-Market": ["exploration score"],
        "Strategic Agility Index": ["managing complexity score"],
        "Strategic Hire Analysis": [],
        "Business Performance": [],
        "Safeguarding Innovation": [],
        "Summary": ["attachment score", "exploration score", "managing complexity score"]
    }

    aggregate_texts = []
    if focus == "Strategic Hire Analysis":
        combined_ratios = (
            f"- Content-Optimisation: {ratio_quadrant_1}\n"
            f"- Relationship-Optimisation: {ratio_quadrant_2}\n"
            f"- Relationship-Exploration: {ratio_quadrant_3}\n"
            f"- Content-Exploration: {ratio_quadrant_4}\n"
            f"- Operational Core: {ratio_quadrant_5}\n"
        )
        aggregate_texts.append(combined_ratios)
    elif focus == "Business Performance":
        combined_ratios = (
            f"- Phase 1 (A&E > 65): {ratio_high_high}\n"
            f"- Phase 2 (A&E 35-65): {ratio_mid_mid}\n"
            f"- Phase 3 (A&E < 35): {ratio_low_low}\n"
        )
        aggregate_texts.append(combined_ratios)
    elif focus == "Safeguarding Innovation":
        combined_ratios = (
            f"- Phase 1 (A < 35 & E > 65): {ratio_low_high}\n"
            f"- Phase 2 (35-65): {ratio_mid_mid}\n"
            f"- Phase 3 (A > 65 & E < 35): {ratio_high_low}\n"
        )
        aggregate_texts.append(combined_ratios)
    else:
        for dim in focus_to_dimensions[focus]:
            agg = aggregate_measures[dim]
            aggregate_texts.append(
                f"{dim.replace('_', ' ').title()}\n"
                f"Mean: {agg['mean']}, Std Dev: {agg['std_dev']}\n"
                f"Below 25: {agg['ratio_below_25']}, 25-50: {agg['ratio_25_50']}, 50-75: {agg['ratio_50_75']}, >75: {agg['ratio_above_75']}"
            )
    return "\n".join(aggregate_texts)

# -----------------------------
# Streamlit UI
# -----------------------------
st.title("AEM-Cube Team Analysis (Debug Mode)")
uploaded_file = st.file_uploader("Upload AEM-Cube Team PDF", type="pdf")

if uploaded_file:
    with st.spinner("Extracting scores and generating analyses..."):
        test_team = extract_scores_from_pdf(uploaded_file)
        if test_team.empty:
            st.error("No scores could be extracted from this PDF.")
        else:
            st.success("Scores extracted successfully.")
            st.dataframe(test_team)

            focus_areas = [
                "Product-Market Fit", "Speed-to-Market", "Strategic Agility Index",
                "Strategic Hire Analysis", "Business Performance", "Safeguarding Innovation", "Summary"
            ]

            for focus in focus_areas:
                st.subheader(focus)
                team_scores = generate_team_score_summary(focus, test_team)
                theory = extract_theory_block(focus, foundational_text)
                example = extract_example_block(focus, description_text)

                prompt = f"""
You are an expert team analyst using the AEM-Cube framework.

### THEORY
```text
{theory}
```

### EXAMPLE
```text
{example}
```

### TEAM SCORES
{team_scores}

Now generate a {focus} analysis. Be concise, logical, and aligned with the theory. Only highlight imbalances if the data shows it. Never contradict the above context.
"""

                token_count = len(encoding.encode(prompt))
                if token_count > 7000:
                    st.warning(f"⚠️ Prompt is {token_count} tokens long. This may cause truncation or hallucinations.")

                with st.expander("🧠 Show Prompt"):
                    st.code(prompt)

                response = client.chat.completions.create(
                    model="gpt-4-1106-preview",
                    messages=[
                        {"role": "system", "content": "You are a structured and insightful team analyst."},
                        {"role": "user", "content": prompt}
                    ],
                    temperature=0.3
                )

                st.markdown(response.choices[0].message.content)
