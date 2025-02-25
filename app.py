import streamlit as st
import requests
import pandas as pd

# Set up your FastAPI endpoint (adjust with your actual endpoint)
API_URL = "http://127.0.0.1:8000/generate-descriptions/"

# Streamlit UI components for inputs
st.title("Project and Team Member Scoring")

# Project name input
project_name = st.text_input("Project Name")

# Team members input (optional, but can be a list of names)
team_members = st.text_area("Team Members (comma-separated)", "")

# Submit button
if st.button("Generate Descriptions"):
    # Prepare data to send to FastAPI backend
    request_data = {}
    if project_name:
        request_data["project_name"] = project_name
    if team_members:
        request_data["team_members"] = [member.strip() for member in team_members.split(",")]

    if request_data:
        # Send the data to the FastAPI backend
        response = requests.post(API_URL, json=request_data)

        if response.status_code == 200:
            data = response.json()

            # Display the results
            st.subheader("Generated Descriptions")
            for description in data.get("descriptions", []):
                st.write(f"**{description['focus']}**")
                st.write(description["description"])

            # Display the team member scores
            st.subheader("Team Member Scores")
            team_member_df = pd.DataFrame(data.get("team_member_scores", []))
            st.dataframe(team_member_df)

        else:
            st.error(f"Error: {response.status_code} - {response.text}")
    else:
        st.warning("Please provide either a project name or team members.")

