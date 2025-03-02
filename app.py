import streamlit as st
import pandas as pd
import numpy as np
import scipy.stats
import openai
import os

# Initialize OpenAI API key
openai.api_key = st.secrets["OPENAI_API_KEY"]
client = openai.Client(api_key=openai.api_key)

MODEL_ID = st.secrets["MODEL_ID"]

# Define focus-to-dimensions mapping
focus_to_dimensions = {
    "Product-Market-Fit": ["attachment score"],
    "Speed-to-Market": ["exploration score"],
    "Strategic Agility Index": ["managing complexity score"],
    "Strategic Hire Analysis": ["attachment score", "exploration score"],
    "Business Performance": ["attachment score", "exploration score"],
    "Safeguarding Innovation": ["exploration score", "attachment score"]
}

# Streamlit UI
st.title("AI Team Analyses")

# Upload CSV file
uploaded_file = st.file_uploader("Upload CSV file", type=["csv"])

# Project name input
project_name = st.text_input("Project Name")

# Team members input (using text_area for free-form input)
team_members_input = st.text_area("Enter Team Members (comma-separated)", "")

# If file is uploaded, process the data
if uploaded_file is not None:
    try:
        # Read the uploaded CSV into a pandas DataFrame
        data = pd.read_csv(uploaded_file)

        # Process team members input (converting to list after splitting by commas)
        team_member_list = [member.strip() for member in team_members_input.split(",")] if team_members_input else []

        # Filter based on project name or team members
        if project_name:
            data = data[data["Project"].str.lower() == project_name.lower()]

        if not project_name and team_member_list:
            data = data[data["Participant"].str.lower().isin([member.lower() for member in team_member_list])]

        if not project_name and not team_member_list:
            st.warning("Please provide either a project name or team members to filter the data.")

        # Filter 'self-image' entries
        data = data[data["Type"] == "self-image"]

        # Remove duplicates
        data = data.drop_duplicates(subset="Participant")

        # Calculate scores
        M_score = pd.concat([data.iloc[:, 8:14], data.iloc[:, 20:26], data.iloc[:, 32:38], data.iloc[:, 44:50]], axis=1)
        E_score = pd.concat([data.iloc[:, 14:20], data.iloc[:, 38:44]], axis=1)
        A_score = pd.concat([data.iloc[:, 26:32], data.iloc[:, 50:56]], axis=1)

        data['attachment score'] = A_score.sum(axis=1)
        data['exploration score'] = E_score.sum(axis=1)
        data['managing complexity score'] = M_score.sum(axis=1)

        # Normalize scores
        data['attachment score'] = np.round(100 * scipy.stats.norm.cdf((data['attachment score'] - 52.6) / 9.0), 0)
        data['exploration score'] = np.round(100 * scipy.stats.norm.cdf((data['exploration score'] - 53.4) / 9.7), 0)
        data['managing complexity score'] = np.round(100 * scipy.stats.norm.cdf((data['managing complexity score'] - 114.6) / 12.4), 0)

        # Calculate aggregate measures
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

        # Prepare to generate descriptions
        descriptions = []
        team_member_scores = []

        # Loop through each focus area and generate a prompt for the API
        for focus, dimensions in focus_to_dimensions.items():
            dimensions_text = ', '.join(dimensions)
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

            # Construct a specific prompt for each focus area
            if focus == "Product-Market-Fit":
                prompt = (
                    f"Generate a detailed description for the team '{project_name}' with focus on the Attachment dimension.\n\n"
                    f"Here are the aggregate measures of the team scores per dimension:\n"
                    f"{'\n'.join(aggregate_texts)}\n\n"
                    f"Provide an analysis that describes how the Attachment score influences the team's alignment between product and customer needs."
                )
            elif focus == "Speed-to-Market":
                prompt = (
                    f"Generate a detailed description for the team '{project_name}' with focus on the Exploration dimension.\n\n"
                    f"Here are the aggregate measures of the team scores per dimension:\n"
                    f"{'\n'.join(aggregate_texts)}\n\n"
                    f"Provide an analysis that describes how the Exploration score impacts the team's speed to market."
                )
            elif focus == "Strategic Agility Index":
                prompt = (
                    f"Generate a detailed description for the team '{project_name}' with focus on the Managing Complexity dimension.\n\n"
                    f"Here are the aggregate measures of the team scores per dimension:\n"
                    f"{'\n'.join(aggregate_texts)}\n\n"
                    f"Provide an analysis that describes how the Managing Complexity score influences the team's ability to manage both complex and complicated problems."
                )
                
            elif focus == "Strategic Hire Analysis":
                prompt = (
                    f"Generate a detailed description for the team '{project_name}' with focus on the Strategic Hire Analysis.\n\n"
                f"Here are the aggregate measures of the team scores per dimension:\n"
                f"{'\n'.join(aggregate_texts)}\n\n"
                f"Provide an analysis that uses the provided Attachment and Exploration scores to categorize individuals into one of the five result areas from the Strategic Hire Analysis: 1. Relationship-Optimization Quadrant (High Attachment score combined with Low Exploration Score), 2.Content-Optimization Quadrant (Low Attachment score with Low Exploration Score) , 3. Relationship-Exploration Quadrant (High Attachment score combined with High Exploration Score), 4. Content-Exploration Quadrant (Low Attachment Score combined with High Exploration score), 5. Strategic Execution Zone (Medium Attachment score combined with medium Exploration score). Based on the categorization, describe the team’s strengths and potential gaps in performance. Be sure to reflect on how the combination of the Attachment and Exploration dimensions shapes each individual’s contribution to the team."
        )
    
            elif focus == "Business Performance":
                prompt = (
                    f"Generate a detailed analysis of the team '{project_name}' with focus on the Business Performance Dialogue. The analysis should consider the team’s contributions in the following phases: 1. Changing with People (High Attachment & High Exploration), 2. Operational Core (Mid Attachment & Mid Exploration), 3. Structured Delivery (Low Attachment & Low Exploration)\n\n"
                    f"Here is the distribution of the teammembers across the three phases:\n"
                    f"{'\n'.join(aggregate_texts)}\n\n"
                    f"Provide an analysis that analyzes the distribution of teammembers across the three phases. Highlights how these contributions shape the team's performance and collaboration in each phase (e.g., driving change, converting ideas into projects, ensuring execution). Points out any gaps or underrepresented contributions in specific phases. Ideally, a well-balanced team has an even distribution of members across all three phases, ensuring the team is equipped to be effective in all areas. Suggests strategies to address these gaps, ensuring smoother transitions across phases."
        )

            elif focus == "Safeguarding Innovation":
                prompt = (
                    f"Generate a detailed analysis of the team '{project_name}' with focus on the Safeguarding Innovation Dialogue.\n\n"
                    f"Here is the distribution of the teammembers across the three phases:\n"
                    f"{'\n'.join(aggregate_texts)}\n\n"
                    f"Provide an analysis that: Identifies how the teammembers are distributed across the three phases: Inventive Exploration, Operational Testing, and Sustaining with People. Highlights any gaps or areas where the team may lack strong contributions. Suggests how the team can better navigate the transition of ideas from research and experimentation to practical, sustainable implementation. Proposes strategies to address these gaps, including possible adjustments to team roles or introducing new perspectives to ensure successful embedding of innovations.  Ideally, a well-balanced team has an even distribution of members across all three phases, ensuring the team is equipped to be effective in all areas."
        )


            # Request from OpenAI API
            response = client.chat.completions.create(
                model=MODEL_ID,  # Replace with your fine-tuned model ID
                messages=[{"role": "system", "content": "You are a helpful assistant."},
                          {"role": "user", "content": prompt}],
                temperature=0.4
            )

            # Accessing the generated content from the response
            generated_content = response.choices[0].message.content.strip()

            # Append the generated content along with the focus to the descriptions list
            descriptions.append({"focus": focus, "description": generated_content})

        # Prepare team member scores
        for _, row in data.iterrows():
            member_info = {
                "name": row["Participant"],
                "attachment_score": row["attachment score"],
                "exploration_score": row["exploration score"],
                "managing_complexity_score": row["managing complexity score"]
            }
            team_member_scores.append(member_info)

        # Display results
        st.subheader("Generated Descriptions")
        for desc in descriptions:
            st.write(f"**{desc['focus']}**:\n{desc['description']}\n")

        # Display team member scores
        st.subheader("Team Member Scores")
        st.dataframe(data[["Participant", "attachment score", "exploration score", "managing complexity score"]])

    except Exception as e:
        st.error(f"An error occurred: {str(e)}")

else:
    st.warning("Please upload a CSV file.")




