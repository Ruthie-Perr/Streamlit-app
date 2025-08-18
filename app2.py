# app.py
import streamlit as st
import pandas as pd
import numpy as np
import openai
import re
import pdfplumber

# ────────────────────────────────────────────
# OpenAI setup
# ────────────────────────────────────────────
openai.api_key = st.secrets["OPENAI_API_KEY"]
client = openai.Client(api_key=openai.api_key)
MODEL_ID = st.secrets["MODEL_ID"]

# ────────────────────────────────────────────
# Focus areas and prompt guidance
# ────────────────────────────────────────────
focus_to_dimensions = {
    "Product-Market-Fit": ["attachment score"],
    "Speed-to-Market": ["exploration score"],
    "Strategic Agility Index": ["managing complexity score"],
    "Strategic Hire Analysis": ["attachment score", "exploration score"],
    "Business Performance": ["attachment score", "exploration score"],
    "Safeguarding Innovation": ["attachment score", "exploration score"],
    "Summary": ["attachment score", "exploration score", "managing complexity score"]
}

focus_prompts = {
    "Summary": "Provide a comprehensive analysis of the team’s distribution across the three AEM-Cube dimensions: Attachment (content vs. relationship focus), Exploration (optimising vs. explorative), and Managing Complexity (specialist vs. generalist). Explain how this distribution shapes the team’s collaboration, working style, and decision-making. Identify key strengths, imbalances, or underrepresented areas, and discuss how these may impact the team’s ability to innovate, deliver, and adapt. Suggest how the team can improve its strategic agility by reinforcing overlooked perspectives, adjusting roles, or developing bridging contributions to support long-term performance and alignment.",
    "Product-Market-Fit": "Provide an analysis of how the Attachment score influences the team's alignment between product and customer needs...",
    "Speed-to-Market": "Analyze how the Exploration score influences the team's ability to balance innovation with the practical demands of delivering solutions in a timely manner...",
    "Strategic Agility Index": "Provide an analysis that describes how the Managing Complexity score influences the team's ability to manage complex problems...",
    "Strategic Hire Analysis": "Provide an analysis that reflects on the distribution of the individuals into the result areas from the Strategic Hire Analysis...",
    "Business Performance": "Provide an analysis that analyzes the distribution of teammembers across the three phases of the Business Performance Dialogue...",
    "Safeguarding Innovation": "Provide an analysis that identifies how the teammembers are distributed across the three phases of the Safeguarding Innovation dialogue..."
}

# ────────────────────────────────────────────
# Helpers: extract final scores from PDF
# ────────────────────────────────────────────
def _last_int(line: str):
    nums = re.findall(r"\d+", line)
    return int(nums[-1]) if nums else None

def extract_scores_from_pdf(pdf_file_like) -> pd.DataFrame:
    rows = []
    with pdfplumber.open(pdf_file_like) as pdf:
        for page in pdf.pages:
            text = page.extract_text() or ""
            if not text.strip():
                continue

            a, e, c = None, None, None
            for raw in text.splitlines():
                ls = raw.strip().lower()
                if ls.startswith("attachment"):
                    a = _last_int(raw)
                elif ls.startswith(("exploration", "exploratie")):
                    e = _last_int(raw)
                elif ls.startswith(("managing", "managen")):
                    c = _last_int(raw)

            if a is not None and e is not None and c is not None:
                rows.append({
                    "attachment score": a,
                    "exploration score": e,
                    "managing complexity score": c,
                })

    return pd.DataFrame(rows)

# ────────────────────────────────────────────
# UI
# ────────────────────────────────────────────
st.title("AI Team Analyses (PDF input)")

uploaded_file = st.file_uploader("Upload PDF file", type=["pdf"])

if uploaded_file is not None:
    try:
        # Extract scores
        data = extract_scores_from_pdf(uploaded_file)

        if data.empty:
            st.error("Error.")
            st.stop()

        # Aggregate measures
        aggregate_measures = {}
        for dim in ["attachment score", "exploration score", "managing complexity score"]:
            scores = data[dim]
            aggregate_measures[dim] = {
                "mean": round(scores.mean(), 2),
                "std_dev": round(scores.std(), 2),
                "ratio_below_25": round((scores < 25).mean(), 2),
                "ratio_25_50": round(((scores >= 25) & (scores < 50)).mean(), 2),
                "ratio_50_75": round(((scores >= 50) & (scores < 75)).mean(), 2),
                "ratio_above_75": round((scores >= 75).mean(), 2)
            }

        # Combined ratios
        ratio_high_high = np.round(((data["attachment score"] > 65) & (data["exploration score"] > 65)).mean(), 2)
        ratio_mid_mid   = np.round(((data["attachment score"].between(35, 65)) & (data["exploration score"].between(35, 65))).mean(), 2)
        ratio_low_low   = np.round(((data["attachment score"] < 35) & (data["exploration score"] < 35)).mean(), 2)
        ratio_low_high  = np.round(((data["attachment score"] < 35) & (data["exploration score"] > 65)).mean(), 2)
        ratio_high_low  = np.round(((data["attachment score"] > 65) & (data["exploration score"] < 35)).mean(), 2)
        ratio_quadrant_1 = np.round(((data["attachment score"] < 51) & (data["exploration score"] < 51)).mean(), 2)
        ratio_quadrant_2 = np.round(((data["attachment score"] > 50) & (data["exploration score"] < 51)).mean(), 2)
        ratio_quadrant_3 = np.round(((data["attachment score"] > 50) & (data["exploration score"] > 50)).mean(), 2)
        ratio_quadrant_4 = np.round(((data["attachment score"] < 51) & (data["exploration score"] > 50)).mean(), 2)
        ratio_quadrant_5 = np.round(((data["attachment score"].between(37.5,62.5)) & (data["exploration score"].between(37.5,62.5))).mean(), 2)

        # Generate analyses
        descriptions = []
        for focus, dimensions in focus_to_dimensions.items():
            aggregate_texts = [
                f"- **{dim}**\n"
                f"  - Mean: {aggregate_measures[dim]['mean']}\n"
                f"  - Std Dev: {aggregate_measures[dim]['std_dev']}\n"
                f"  - Ratio <25: {aggregate_measures[dim]['ratio_below_25']}\n"
                f"  - Ratio 25-50: {aggregate_measures[dim]['ratio_25_50']}\n"
                f"  - Ratio 50-75: {aggregate_measures[dim]['ratio_50_75']}\n"
                f"  - Ratio >75: {aggregate_measures[dim]['ratio_above_75']}"
                for dim in dimensions
            ]

            if focus == "Strategic Hire Analysis":
                aggregate_texts.extend([
                    f"- Quadrant 1 Content-Optimisation: {ratio_quadrant_1}",
                    f"- Quadrant 2 Relationship-Optimisation: {ratio_quadrant_2}",
                    f"- Quadrant 3 Relationship-Exploration: {ratio_quadrant_3}",
                    f"- Quadrant 4 Content-Exploration: {ratio_quadrant_4}",
                    f"- Quadrant 5 Operational Core: {ratio_quadrant_5}",
                ])
            elif focus == "Business Performance":
                aggregate_texts.extend([
                    f"- Phase 1 High-High: {ratio_high_high}",
                    f"- Phase 2 Mid-Mid: {ratio_mid_mid}",
                    f"- Phase 3 Low-Low: {ratio_low_low}"
                ])
            elif focus == "Safeguarding Innovation":
                aggregate_texts.extend([
                    f"- Phase 1 Low-High: {ratio_low_high}",
                    f"- Phase 2 Mid-Mid: {ratio_mid_mid}",
                    f"- Phase 3 High-Low: {ratio_high_low}"
                ])

            prompt = (
                f"Generate a detailed description for the team with focus on {', '.join(dimensions)}.\n\n"
                f"Here are the aggregates:\n{'\n'.join(aggregate_texts)}\n\n"
                f"{focus_prompts.get(focus, '')}"
            )

            response = client.chat.completions.create(
                model=MODEL_ID,
                messages=[
                    {"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.3
            )
            descriptions.append({"focus": focus, "description": response.choices[0].message.content.strip()})

        # Output
        st.subheader("Generated Descriptions")
        for d in descriptions:
            st.write(f"**{d['focus']}**:\n{d['description']}\n")

        st.subheader("Team Member Scores")
        st.dataframe(data)

    except Exception as e:
        st.error(f"Error: {e}")
else:
    st.warning("Please upload a PDF file.")







