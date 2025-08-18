# app.py
import streamlit as st
import pandas as pd
import numpy as np
import scipy.stats
import openai
import re
import pdfplumber
import os
import json

# ──────────────────────────────────────────────────────────────────────────────
# OpenAI setup (expects OPENAI_API_KEY and MODEL_ID in Streamlit Secrets)
# ──────────────────────────────────────────────────────────────────────────────
openai.api_key = st.secrets["OPENAI_API_KEY"]
client = openai.Client(api_key=openai.api_key)
MODEL_ID = st.secrets["MODEL_ID"]

# ──────────────────────────────────────────────────────────────────────────────
# Focus areas and prompt guidance (unchanged)
# ──────────────────────────────────────────────────────────────────────────────
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
    "Product-Market-Fit": "Provide an analysis of how the Attachment score (ranging from 0 to 100) influences the team's alignment between product and customer needs. A low Attachment score (0-40) indicates a stronger focus on product and technical content, prioritizing internal development over direct customer engagement, while a high Attachment score (60-100) suggests a team that is highly customer-centric, prioritizing relationships and market needs. A medium Attachment score (40-60) represents a balance between these two approaches. Discuss how the distribution of Attachment scores affects the team’s ability to achieve market fit, customer engagement, and responsiveness to user needs, highlighting potential gaps in understanding customer perspectives. A well-balanced team is not one where everyone has similar scores but rather one with a diverse mix of low, medium, and high scores, ensuring a variety of perspectives and capabilities. First, summarize the distribution of scores. Then, based on this, analyze the team’s alignment and blind spots while ensuring logical consistency.",
    "Speed-to-Market": "Analyze how the Exploration score influences the team's ability to balance innovation with the practical demands of delivering solutions in a timely manner. This dialogue assesses how well a team balances innovation and efficiency in R&D, product development, and regulatory processes. It explores whether the team leans more toward Exploration (high Exploration scores >60)—indicating a strong focus on innovation, experimentation, and discovering new possibilities—or toward Optimization (low Exploration scores <40)—indicating a focus on refining, streamlining, and efficiently executing established processes. A well-balanced team is not one where all members cluster in the middle (Exploration score between 40 and 60) but rather one where there is a comparable number of members across the low, medium, and high score ranges. This distribution ensures diversity in perspectives and capabilities, allowing for both ambitious innovation and pragmatic execution. First, summarize the distribution of scores. Then, based on this, analyze the team’s alignment and blind spots while ensuring logical consistency.",
    "Strategic Agility Index": "Provide an analysis that describes how the Managing Complexity score influences the team's ability to manage complex problems, based on the logic that a team needs both specialists and generalists to manage complexity well. Consider how the score affects the team's adaptability, decision-making, and overall strategic execution. A well-balanced team has an even distribution of members across the score quadrants, rather than clustering in one or two areas. This means that team members should be spread across low, medium, and high ranges of scores rather than all scoring similarly. Each quadrant should have a comparable number of members to ensure diversity in perspectives and capabilities. First, summarize the distribution of scores. Then, based on this, analyze the team’s alignment and blind spots while ensuring logical consistency.",
    "Strategic Hire Analysis": "Provide an analysis that reflects on the distribution of the individuals into the result areas from the Strategic Hire Analysis: 1. Relationship-Optimization Quadrant (Attachment score >50 combined with Exploration Score <50), 2.Content-Optimization Quadrant (Attachment score <50 with Exploration Score <50) , 3. Relationship-Exploration Quadrant (Attachment score >50 combined with Exploration Score> 50), 4. Content-Exploration Quadrant (Attachment Score<50 combined with Exploration score>50), 5. Operational Core (Medium Attachment score combined with medium Exploration score). Based on the categorization, describe the team’s strengths and  potential gaps in performance. Be sure to reflect on how the combination of the Attachment and Exploration dimensions shapes each individual’s contribution to the team. Ideally, the team has a nice spread across the different result-areas. When there is an imbalance in these quadrants (e.g., over or under-representation in one area), suggest potential consequences or blind spots, and provide recommendations for balancing the team.",
    "Business Performance": "Provide an analysis that analyzes the distribution of teammembers across the three phases of the Business Performance Dialogue - Phase 1: Changing with People (high attachment and high exploration), Phase 2: Operational Core (mid attachment and mid exploration), phase 3: Structured Delivery (low attachment and low exploration). Highlights how these contributions shape the team's performance and collaboration in each phase.. Points out any gaps or underrepresented contributions in specific phases. Ideally, a well-balanced team has an even distribution of members across all three phases, ensuring the team is equipped to be effective in all areas. Suggests strategies to address these gaps, ensuring smoother transitions across phases.",
    "Safeguarding Innovation": "Provide an analysis that: Identifies how the teammembers are distributed across the three phases of the Safeguarding Innovaiton dialogue: 1. Inventive Exploration (low attachment and high exploration), 2. Operational Testing (mid attachment and mid exploration), and 3. Sustaining with People (high attachment and low exploration). Highlights any gaps or areas where the team may lack strong contributions. Suggests how the team can better navigate the transition of ideas from research and experimentation to practical, sustainable implementation. Proposes strategies to address these gaps, including possible adjustments to team roles or introducing new perspectives to ensure successful embedding of innovations.  Ideally, a well-balanced team has an even distribution of members across all three phases, ensuring the team is equipped to be effective in all areas."
}

# ──────────────────────────────────────────────────────────────────────────────
# Helpers: extract final scores from PDF (per page = one participant)
# ──────────────────────────────────────────────────────────────────────────────
def _last_int(line: str):
    nums = re.findall(r"\d+", line)
    return int(nums[-1]) if nums else None

def extract_scores_from_pdf(pdf_file_like, project_name: str = "Team zelfbeelden") -> pd.DataFrame:
    """
    Parses a PDF where each page contains one participant's results.
    Looks for lines starting with 'Participant:'/'Name:', and lines starting with
    'Attachment', 'Exploration/Exploratie', 'Managing/Managen'.
    Returns a DataFrame with columns:
      Participant, Project, Type, attachment score, exploration score, managing complexity score
    """
    rows = []
    with pdfplumber.open(pdf_file_like) as pdf:
        for page in pdf.pages:
            text = page.extract_text() or ""
            if not text.strip():
                continue

            participant = None
            a = e = c = None

            for raw in text.splitlines():
                ls = raw.strip()
                lsl = ls.lower()

                # Participant name
                if lsl.startswith("participant:") or lsl.startswith("name:"):
                    participant = ls.split(":", 1)[-1].strip()

                # Scores (final, ready to use)
                elif lsl.startswith("attachment"):
                    v = _last_int(ls);  a = v if v is not None else a
                elif lsl.startswith(("exploration", "exploratie")):
                    v = _last_int(ls);  e = v if v is not None else e
                elif lsl.startswith(("managing", "managen")):
                    v = _last_int(ls);  c = v if v is not None else c

            if a is not None and e is not None and c is not None:
                rows.append({
                    "Project": project_name,
                    "Participant": participant or f"Unknown_{len(rows)+1}",
                    "attachment score": a,
                    "exploration score": e,
                    "managing complexity score": c,
                })

    return pd.DataFrame(rows)

# ──────────────────────────────────────────────────────────────────────────────
# UI
# ──────────────────────────────────────────────────────────────────────────────
st.title("AI Team Analyses")

uploaded_file = st.file_uploader("Upload PDF file", type=["pdf"])
project_name = st.text_input("Project Name")
team_members_input = st.text_area("Enter Team Members (comma-separated)", "")

# ──────────────────────────────────────────────────────────────────────────────
# Main flow (PDF only)
# ──────────────────────────────────────────────────────────────────────────────
if uploaded_file is not None:
    try:
        # Extract final scores from PDF (no extra calculations/normalization)
        data = extract_scores_from_pdf(uploaded_file, project_name or "Team zelfbeelden")

        # Process team members input
        team_member_list = [m.strip() for m in team_members_input.split(",")] if team_members_input else []

        # Team Member validation
        if team_member_list:
            if "Participant" not in data.columns or data.empty:
                st.error("No participants found in the PDF.")
                st.stop()
            available_members = data["Participant"].str.lower().unique()
            entered_members = [m.lower() for m in team_member_list]
            missing_members = [m for m in entered_members if m not in available_members]
            if missing_members:
                st.error(f"The following team members were not found in the data: {', '.join(missing_members)}")
                st.stop()

        # Filter based on project name and/or team members
        if project_name and team_member_list:
            data = data[data["Project"].str.lower() == project_name.lower()]
            data = data[data["Participant"].str.lower().isin([m.lower() for m in team_member_list])]
        elif project_name:
            data = data[data["Project"].str.lower() == project_name.lower()]
        elif team_member_list:
            data = data[data["Participant"].str.lower().isin([m.lower() for m in team_member_list])]
        elif not project_name and not team_member_list:
            st.warning("Please provide either a project name or team members to filter the data.")

        # Keep only self-image entries (we set this in extractor)
        if "Type" in data.columns:
            data = data[data["Type"].str.lower() == "self-image"]

        # Remove duplicates per participant
        data = data.drop_duplicates(subset="Participant")

        # At this point the scores are final and already in the required columns:
        #   'attachment score', 'exploration score', 'managing complexity score'
        # No normalization here (you requested to keep PDF scores as-is)

        # Calculate aggregate measures (unchanged)
        aggregate_measures = {}
        for dimension in ["attachment score", "exploration score", "managing complexity score"]:
            scores = data[dimension]
            aggregate_measures[dimension] = {
                "mean": round(scores.mean(), 2),
                "std_dev": round(scores.std(), 2),
                "ratio_below_25": round((scores < 25).mean(), 2),
                "ratio_25_50": round(((scores >= 25) & (scores < 50)).mean(), 2),
                "ratio_50_75": round(((scores >= 50) & (scores < 75)).mean(), 2),
                "ratio_above_75": round((scores >= 75).mean(), 2)
            }

        # Combined ratios (unchanged)
        ratio_high_high = np.round(((data["attachment score"] > 65) & (data["exploration score"] > 65)).mean(), 2)
        ratio_mid_mid   = np.round(((data["attachment score"].between(35, 65)) & (data["exploration score"].between(35, 65))).mean(), 2)
        ratio_low_low   = np.round(((data["attachment score"] < 35) & (data["exploration score"] < 35)).mean(), 2)
        ratio_low_high  = np.round(((data["attachment score"] < 35) & (data["exploration score"] > 65)).mean(), 2)
        ratio_mid_mid_2 = np.round(((data["attachment score"].between(35, 65)) & (data["exploration score"].between(35, 65))).mean(), 2)
        ratio_high_low  = np.round(((data["attachment score"] > 65) & (data["exploration score"] < 35)).mean(), 2)
        ratio_quadrant_1 = np.round(((data["attachment score"] < 51) & (data["exploration score"] < 51)).mean(), 2)
        ratio_quadrant_2 = np.round(((data["attachment score"] > 50) & (data["exploration score"] < 51)).mean(), 2)
        ratio_quadrant_3 = np.round(((data["attachment score"] > 50) & (data["exploration score"] > 50)).mean(), 2)
        ratio_quadrant_4 = np.round(((data["attachment score"] < 51) & (data["exploration score"] > 50)).mean(), 2)
        ratio_quadrant_5 = np.round(((data["attachment score"] >= 37.5) & (data["attachment score"] <= 62.5) &
                                     (data["exploration score"] >= 37.5) & (data["exploration score"] <= 62.5)).mean(), 2)

        # Prepare to generate descriptions (unchanged)
        descriptions = []

        for focus, dimensions in focus_to_dimensions.items():
            aggregate_texts = [
                f"- **{dim.replace('_', ' ').title()}**\n"
                f"  - Mean: {aggregate_measures[dim]['mean']}\n"
                f"  - Standard Deviation: {aggregate_measures[dim]['std_dev']}\n"
                f"  - Ratio below 25: {aggregate_measures[dim]['ratio_below_25']}\n"
                f"  - Ratio 25-50: {aggregate_measures[dim]['ratio_25_50']}\n"
                f"  - Ratio 50-75: {aggregate_measures[dim]['ratio_50_75']}\n"
                f"  - Ratio above 75: {aggregate_measures[dim]['ratio_above_75']}"
                for dim in dimensions
            ]

            if focus == "Strategic Hire Analysis":
                combined_texts = [
                    f"- Ratio of team members in the Content-Optimisation quadrant: {ratio_quadrant_1}",
                    f"- Ratio of team members in the Relationship-Optimisation quadrant: {ratio_quadrant_2}",
                    f"- Ratio of team members in the Relationship-Exploration quadrant: {ratio_quadrant_3}",
                    f"- Ratio of team members in the Content-Exploration quadrant: {ratio_quadrant_4}",
                    f"- Ratio of team members in the Operational Core: {ratio_quadrant_5}",
                ]
            elif focus == "Business Performance":
                combined_texts = [
                    f"- Ratio of team members in Phase 1: with high Attachment and Exploration > 65: {ratio_high_high}",
                    f"- Ratio of team members in Phase 2: with Attachment & Exploration between 35-65: {ratio_mid_mid}",
                    f"- Ratio of team members in Phase 3: with Attachment & Exploration < 35: {ratio_low_low}"
                ]
            elif focus == "Safeguarding Innovation":
                combined_texts = [
                    f"- Ratio of team members in Phase 1: with Attachment < 35 & Exploration > 65: {ratio_low_high}",
                    f"- Ratio of team members in Phase 2: with Attachment between 35-65 & Exploration between 35-65: {ratio_mid_mid_2}",
                    f"- Ratio of team members in Phase 3:with Attachment > 65 & Exploration < 35: {ratio_high_low}"
                ]
            else:
                combined_texts = []

            aggregate_texts.extend(combined_texts)

            prompt = (
                f"Generate a detailed description for the team '{project_name}' with focus on the following score dimensions: {', '.join(dimensions)}.\n\n"
                f"Here are the aggregate measures of the team scores per dimension:\n"
                f"{'\n'.join(aggregate_texts)}\n\n"
                f"{focus_prompts.get(focus, '')}\n\n"
                f"Provide an analysis that incorporates these scores and ratios to describe the team's performance and potential areas for improvement."
            )

            response = client.chat.completions.create(
                model=MODEL_ID,
                messages=[
                    {"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.3
            )
            generated_content = response.choices[0].message.content.strip()
            descriptions.append({"focus": focus, "description": generated_content})

        # Display results (unchanged)
        st.subheader("Generated Descriptions")
        for desc in descriptions:
            st.write(f"**{desc['focus']}**:\n{desc['description']}\n")

        st.subheader("Team Member Scores")
        st.dataframe(data[["Participant", "attachment score", "exploration score", "managing complexity score"]])

    except Exception as e:
        st.error(f"An error occurred: {str(e)}")
else:
    st.warning("Please upload a PDF file.")

