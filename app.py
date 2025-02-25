import streamlit as st
import pandas as pd
import numpy as np
import scipy.stats
import openai
import os
from openai import OpenAI


import os


# Get the API key and model ID from the environment
openai.api_key = st.secrets["OPENAI_API_KEY"]
client = OpenAI(api_key=os.st.secrets["OPENAI_API_KEY"])

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
st.title("Project and Team Member Scoring")

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

        # If no project name is provided but team members are given, filter by team members
        if not project_name and team_member_list:
            # Convert the "Participant" column to lowercase for case-insensitive matching
            data = data[data["Participant"].str.lower().isin([member.lower() for member in team_member_list])]

       # If neither project name nor team members are provided, show a warning
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

            prompt = (
                f"Generate a description for the team '{project_name}' with focus on the following dimension(s): {dimensions_text}.\n\n"
                f"Here are the aggregate measures of the team scores:\n"
                f"{'\n'.join(aggregate_texts)}\n\n"
                f"Provide an analysis that describes how the team's scores in these areas influence the project."
            )

            # Request from OpenAI API using client
            response = client.chat.completions.create(
                model=MODEL_ID,  # Replace with your fine-tuned model ID
                messages=[
                    {"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.4  # from 0 (very deterministic) to 2 (maximum randomness)
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



