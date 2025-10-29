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
        "Product-Market-Fit": ("1. Product-Market-Fit", "2. Speed-to-Market"),
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

def compute_team_stats(test_team):
    dims = ["attachment score", "exploration score", "managing complexity score"]

    metrics = {}
    proportions = {}
    for dim in dims:
        s = test_team[dim]
        metrics[dim] = {
            "mean": float(np.round(s.mean(), 2)),
            "std_dev": float(np.round(s.std(ddof=0), 2)),  # population SD for stability; change to ddof=1 if you prefer sample SD
        }
        proportions[dim] = {
            "ratio_below_25": float(np.round((s < 25).mean(), 2)),
            "ratio_25_50": float(np.round(((s >= 25) & (s < 50)).mean(), 2)),
            "ratio_50_75": float(np.round(((s >= 50) & (s < 75)).mean(), 2)),
            "ratio_above_75": float(np.round((s >= 75).mean(), 2)),
        }

    # Cross-dimension combined ratios (A vs E) for special focuses
    A = test_team["attachment score"]
    E = test_team["exploration score"]
    combined = {
        "ratio_high_high": float(np.round(((A > 65) & (E > 65)).mean(), 2)),
        "ratio_mid_mid":   float(np.round((A.between(35, 65) & E.between(35, 65)).mean(), 2)),
        "ratio_low_low":   float(np.round(((A < 35) & (E < 35)).mean(), 2)),
        "ratio_low_high":  float(np.round(((A < 35) & (E > 65)).mean(), 2)),
        "ratio_high_low":  float(np.round(((A > 65) & (E < 35)).mean(), 2)),
        "ratio_quadrant_1": float(np.round(((A < 51) & (E < 51)).mean(), 2)),  # Content-Optimisation
        "ratio_quadrant_2": float(np.round(((A > 50) & (E < 51)).mean(), 2)),  # Relationship-Optimisation
        "ratio_quadrant_3": float(np.round(((A > 50) & (E > 50)).mean(), 2)),  # Relationship-Exploration
        "ratio_quadrant_4": float(np.round(((A < 51) & (E > 50)).mean(), 2)),  # Content-Exploration
        "ratio_quadrant_5": float(np.round((A.between(37.5, 62.5) & E.between(37.5, 62.5)).mean(), 2)),  # Operational Core
    }

    return {"metrics": metrics, "proportions": proportions, "combined": combined}


def format_team_sections(focus, stats):
    focus_to_dimensions = {
        "Product-Market-Fit": ["attachment score"],
        "Speed-to-Market": ["exploration score"],
        "Strategic Agility Index": ["managing complexity score"],
        "Strategic Hire Analysis": [],  # uses combined
        "Business Performance": [],     # uses combined
        "Safeguarding Innovation": [],  # uses combined
        "Summary": ["attachment score", "exploration score", "managing complexity score"]
    }

    dims = focus_to_dimensions[focus]
    metrics_lines = []
    props_lines = []

    # Always format metrics & proportions when dimensions are defined
    for dim in dims:
        m = stats["metrics"][dim]
        p = stats["proportions"][dim]
        title = dim.replace("_", " ").title()
        metrics_lines.append(f"- {title}: mean={m['mean']}, sd={m['std_dev']}")
        props_lines.append(
            f"- {title}: <25={p['ratio_below_25']}, 25-50={p['ratio_25_50']}, 50-75={p['ratio_50_75']}, >75={p['ratio_above_75']}"
        )

    metrics_block = "\n".join(metrics_lines) if metrics_lines else "(not applicable)"
    proportions_block = "\n".join(props_lines) if props_lines else "(not applicable)"

    # Optional combined section for the special focuses
    combined_block = ""
    if focus == "Strategic Hire Analysis":
        c = stats["combined"]
        combined_block = (
            "Content-Optimisation: {q1}\n"
            "Relationship-Optimisation: {q2}\n"
            "Relationship-Exploration: {q3}\n"
            "Content-Exploration: {q4}\n"
            "Operational Core: {q5}"
        ).format(
            q1=c["ratio_quadrant_1"],
            q2=c["ratio_quadrant_2"],
            q3=c["ratio_quadrant_3"],
            q4=c["ratio_quadrant_4"],
            q5=c["ratio_quadrant_5"],
        )
    elif focus == "Business Performance":
        c = stats["combined"]
        combined_block = (
            "Phase 1 (A&E > 65): {hh}\n"
            "Phase 2 (A&E 35-65): {mm}\n"
            "Phase 3 (A&E < 35): {ll}"
        ).format(hh=c["ratio_high_high"], mm=c["ratio_mid_mid"], ll=c["ratio_low_low"])
    elif focus == "Safeguarding Innovation":
        c = stats["combined"]
        combined_block = (
            "Phase 1 (A < 35 & E > 65): {lh}\n"
            "Phase 2 (35-65): {mm}\n"
            "Phase 3 (A > 65 & E < 35): {hl}"
        ).format(lh=c["ratio_low_high"], mm=c["ratio_mid_mid"], hl=c["ratio_high_low"])

    return metrics_block, proportions_block, combined_block

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
                "Product-Market-Fit", "Speed-to-Market", "Strategic Agility Index",
                "Strategic Hire Analysis", "Business Performance", "Safeguarding Innovation", "Summary"
            ]

            # NEW: compute stats once
            stats = compute_team_stats(test_team)

            for focus in focus_areas:
                st.subheader(focus)

                # NEW: separate blocks for metrics, proportions, and combined ratios
                metrics_text, proportions_text, combined_text = format_team_sections(focus, stats)

                theory = extract_theory_block(focus, foundational_text)
                example = extract_example_block(focus, description_text)

                # NEW: clearly separated sections in the prompt
                prompt = f"""
You are an expert team analyst using the AEM-Cube framework.

### THEORY
<text>
{theory}
</text>

### EXAMPLE
<text>
{example}
</text>

### TEAM METRICS (Means & Standard Deviations)
{metrics_text}

### TEAM DISTRIBUTION (Proportion of Members per Score Range)
{proportions_text}

{('### RATIOS (A × E)\\n' + combined_text) if combined_text else ''}

Now generate a {focus} analysis using the framework of the example in both length of the description and approach, but for the scores provided. Be concise, logical, and aligned with the theory.
Only highlight imbalances if the data shows it. Never contradict the above context.

### HARD CONSTRAINTS
-Write **no more than 250 words**. This is a hard limit, **unless the focus is Summary**, for which the maximum is **500 words**.
- Match the **tone and structure** of the example (length, paragraphing, level of detail).
- Be concise, logical, and aligned with the THEORY only. Do not introduce concepts outside the provided THEORY.
- Output **only** the analysis text (no extra headings or labels).
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
