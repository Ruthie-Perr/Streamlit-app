import streamlit as st
import requests
import pandas as pd
from io import BytesIO


# Set the FastAPI backend URL
API_URL = "http://127.0.0.1:8000/generate-descriptions/"

# Streamlit UI components for inputs
st.title("Project and Team Member Scoring")

# Upload CSV file
uploaded_file = st.file_uploader("Upload CSV file", type=["csv"])

# Project name input
project_name = st.text_input("Project Name")

# Team members input (optional, but can be a list of names)
team_members = st.text_area("Team Members (comma-separated)", "")

# If the user uploads a CSV file and clicks the button
if uploaded_file is not None and st.button("Generate Descriptions"):
    try:
        # Prepare data to send to FastAPI backend
        request_data = {
            "project_name": project_name,
            "team_members": [member.strip() for member in team_members.split(",")] if team_members else [],
        }

        # Send the CSV file and additional data to the backend
        files = {"file": uploaded_file}
        response = requests.post(API_URL, data=request_data, files=files)

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
    
    except Exception as e:
        st.error(f"Error: {str(e)}")

else:
    st.warning("Please upload a CSV file and provide either a project name or team members.")
