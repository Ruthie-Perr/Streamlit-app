import streamlit as st
import fitz
import pandas as pd
import numpy as np
import openai
import re

# -----------------------------
# Configuration
# -----------------------------
openai.api_key = st.secrets["OPENAI_API_KEY"]
client = openai.Client(api_key=openai.api_key)

MODEL_ID = st.secrets["MODEL_ID"]

# -----------------------------
# Functions
# -----------------------------
def extract_scores_from_pdf(pdf_file):
    doc = fitz.open(stream=pdf_file.read(), filetype="pdf")
    scores = []
    for page in doc:
        lines = page.get_text("text").splitlines()
        for i in range(len(lines) - 5):
            line0 = lines[i].strip().lower()
            line1 = lines[i+1].strip().lower()
            line2 = lines[i+2].strip().lower()
            if (
                line0 == "attachment"
                and line1 in ["exploratie", "exploration"]
                and (line2.startswith("managen") or line2.startswith("managing"))
            ):
                try:
                    a = int(lines[i+3].strip())
                    e = int(lines[i+4].strip())
                    c = int(lines[i+5].strip())
                    scores.append({
                        "Project": "Uploaded PDF",
                        "attachment score": a,
                        "exploration score": e,
                        "managing complexity score": c
                    })
                except ValueError:
                    continue
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
            f"- **Ratio of team members in the Content-Optimisation quadrant**: {ratio_quadrant_1}\n"
            f"- **Ratio of team members in the Relationship-Optimisation quadrant**: {ratio_quadrant_2}\n"
            f"- **Ratio of team members in the Relationship-Exploration quadrant**: {ratio_quadrant_3}\n"
            f"- **Ratio of team members in the Content-Exploration quadrant**: {ratio_quadrant_4}\n"
            f"- **Ratio of team members in the Operational Core**: {ratio_quadrant_5}\n"
        )
        aggregate_texts.append(f"Combined Ratios for Strategic Hire Analysis:\n{combined_ratios}")
    elif focus == "Business Performance":
        combined_ratios = (
            f"- **Ratio of team members in Phase 1: with Attachment & Exploration > 65**: {ratio_high_high}\n"
            f"- **Ratio of team members in Phase 2: with Attachment & Exploration between 35-65**: {ratio_mid_mid}\n"
            f"- **Ratio of team members in Phase 3: with Attachment & Exploration < 35**: {ratio_low_low}\n"
        )
        aggregate_texts.append(f"Combined Ratios for Business Performance:\n{combined_ratios}")
    elif focus == "Safeguarding Innovation":
        combined_ratios = (
            f"- **Ratio of team members in Phase 1: with Attachment < 35 & Exploration > 65**: {ratio_low_high}\n"
            f"- **Ratio of team members in Phase 2: with Attachment between 35-65 & Exploration between 35-65**: {ratio_mid_mid}\n"
            f"- **Ratio of team members in Phase 3: with Attachment > 65 & Exploration < 35**: {ratio_high_low}\n"
        )
        aggregate_texts.append(f"Combined Ratios for Safeguarding Innovation:\n{combined_ratios}")
    else:
        for dim in focus_to_dimensions[focus]:
            agg = aggregate_measures[dim]
            aggregate_texts.append(
                f"- **{dim.replace('_', ' ').title()}**\n"
                f"  - Mean: {agg['mean']}\n"
                f"  - Standard Deviation: {agg['std_dev']}\n"
                f"  - Ratio below 25: {agg['ratio_below_25']}\n"
                f"  - Ratio 25-50: {agg['ratio_25_50']}\n"
                f"  - Ratio 50-75: {agg['ratio_50_75']}\n"
                f"  - Ratio above 75: {agg['ratio_above_75']}"
            )
    return "\n".join(aggregate_texts)

# -----------------------------
# Streamlit UI
# -----------------------------
st.title("AEM-Cube Team Analysis")
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

                prompt = f"""
You are an expert team analyst using the AEM-Cube framework.

Use the following score data to write a cohesive {focus} analysis.

Team Score Summary:
{team_scores}

Be concise (2–3 paragraphs). Only describe imbalances or risks if they appear in the score profile. If the team appears balanced, state that clearly and avoid inventing blind spots. Conclude with a recommendation only if meaningful.

Your goal is to deliver a professional, high-value insight that flows logically and is easy to understand.
"""

                response = openai.ChatCompletion.create(
                    model="gpt-4-1106-preview",
                    messages=[
                        {"role": "system", "content": "You are a structured and insightful team analyst."},
                        {"role": "user", "content": prompt}
                    ],
                    temperature=0.3
                )

                analysis = response["choices"][0]["message"]["content"]
                st.markdown(analysis)
