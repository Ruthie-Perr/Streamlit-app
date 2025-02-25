from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Optional, List

import openai
import pandas as pd
import scipy.stats
import json
import numpy as np

from openai import OpenAI

import os
from dotenv import load_dotenv

# Load environment variables from a .env file
load_dotenv()

# Get the API key and model ID from the environment
openai.api_key = os.getenv("OPENAI_API_KEY")
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))



# Initialize FastAPI
app = FastAPI()

# Define the request model to accept team members and/or project name
class RequestData(BaseModel):
    project_name: Optional[str] = None  # Make project_name optional
    team_members: Optional[List[str]] = None  # Make team_members optional

# Define focus-to-dimensions mapping, ensuring each dialogue gets the correct focus
focus_to_dimensions = {
    "Product-Market-Fit": ["attachment score"],  # Focus on Attachment dimension
    "Speed-to-Market": ["exploration score"],  # Focus on Exploration dimension
    "Strategic Agility Index": ["managing complexity score"],  # Focus on Managing Complexity dimension
    "Strategic Hire Analysis": ["attachment score", "exploration score"],  # Focus on Attachment + Exploration dimensions
    "Business Performance": ["attachment score", "exploration score"],  # Focus on Attachment + Exploration dimensions
    "Safeguarding Innovation": ["exploration score", "attachment score"]  # Focus on Exploration + Attachment dimensions
}


@app.post("/generate-descriptions/")
async def generate_descriptions(request_data: RequestData):
    project_name = request_data.project_name
    team_members = request_data.team_members
    
    # Check if both fields are missing
    if not project_name and not team_members:
        raise HTTPException(status_code=400, detail="Either project_name or team_members must be provided.")
    
    # Load project data from a CSV file (replace with your actual CSV file path)
    try:
        project_data = pd.read_csv("team_data.csv")  # Replace with your actual file path
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error loading data: {str(e)}")
    
    # Filter the data based on the project_name or team_members
    if project_name:
        project_data = project_data[project_data["Project"] == project_name]
    if team_members:
        project_data = project_data[project_data["Participant"].isin(team_members)]

    # Filter to include only rows where 'Type' is 'self-image'
    project_data = project_data[project_data["Type"] == "self-image"]
    
    # Remove duplicate participants, keeping only the first occurrence
    project_data = project_data.drop_duplicates(subset="Participant")
    
    # Check if we found any matching data
    if project_data.empty:
        raise HTTPException(status_code=404, detail="No matching data found for the provided project or team members.")

    # Define the columns for calculating the scores
    M_score = pd.concat([project_data.iloc[:, 8:14], project_data.iloc[:, 20:26], project_data.iloc[:, 32:38], project_data.iloc[:, 44:50]], axis=1)
    E_score = pd.concat([project_data.iloc[:, 14:20], project_data.iloc[:, 38:44]], axis=1)
    A_score = pd.concat([project_data.iloc[:, 26:32], project_data.iloc[:, 50:56]], axis=1)

    # Sum the selected columns row-wise and create new columns with the result
    project_data['attachment score'] = A_score.sum(axis=1)
    project_data['exploration score'] = E_score.sum(axis=1)
    project_data['managing complexity score'] = M_score.sum(axis=1)

    # Transform to normed scores
    project_data['attachment score'] = np.round(100 * scipy.stats.norm.cdf((project_data['attachment score'] - 52.6) / 9.0),0)
    project_data['exploration score'] = np.round(100 * scipy.stats.norm.cdf((project_data['exploration score'] - 53.4) / 9.7),0)
    project_data['managing complexity score'] = np.round(100 * scipy.stats.norm.cdf((project_data['managing complexity score'] - 114.6) / 12.4),0)

    # Compute aggregate measures for each dimension
    aggregate_measures = {}
    for dimension in ["attachment score", "exploration score", "managing complexity score"]:
        scores = project_data[dimension]
        aggregate_measures[dimension] = {
            "mean": round(scores.mean(), 2),
            "std_dev": round(scores.std(), 2),
            "ratio_below_25": round((scores < 25).mean(), 2),
            "ratio_25_50": round(((scores >= 25) & (scores < 50)).mean(), 2),
            "ratio_50_75": round(((scores >= 50) & (scores < 75)).mean(), 2),
            "ratio_above_75": round((scores >= 75).mean(), 2)
        }

    # Generate specific descriptions for each focus area
    descriptions = []
    team_member_scores = []  # To store the scores for each team member

    
    for focus, dimensions in focus_to_dimensions.items():
        dimensions_text = ', '.join(dimensions)
        
        # Format aggregate measures per dimension
        aggregate_texts = [
            f"- **{dim.replace('_', ' ').title()}**\n"
            f"  - Mean: {aggregate_measures[dim]['mean']}\n"
            f"  - Standard Deviation: {aggregate_measures[dim]['std_dev']}\n"
            f"  - Ratio of team members with scores <25: {aggregate_measures[dim]['ratio_below_25']}\n"
            f"  - Ratio of team members with scores 25-50: {aggregate_measures[dim]['ratio_25_50']}\n"
            f"  - Ratio of team members with scores 50-75: {aggregate_measures[dim]['ratio_50_75']}\n"
            f"  - Ratio of team members with score >75: {aggregate_measures[dim]['ratio_above_75']}"
            for dim in dimensions
        ]
        
        # Custom prompts for different dialogues
        if focus == "Product-Market-Fit":
            prompt = (
                f"Generate a detailed description for the team '{project_name}' with focus on the Attachment dimension.\n\n"
                f"Here are the aggregate measures of the team scores per dimension:\n"
                f"{'\n'.join(aggregate_texts)}\n\n"
                f"Provide an analysis that describes how the Attachment score influences the team's alignment between product and customer needs. Discuss the impact of the attachment scores on market-fit, customer engagement, and potential gaps in understanding customer perspectives. Ideally a team is well-balanced, meaning that they have approximately equal ratios members in each of the score quadrants."
            )
        
        elif focus == "Speed-to-Market":
            prompt = (
                f"Generate a detailed description for the team '{project_name}' with focus on the Exploration dimension.\n\n"
                f"Here are the aggregate measures of the team scores per dimension:\n"
                f"{'\n'.join(aggregate_texts)}\n\n"
                f"Provide an analysis that describes how the Exploration score impacts the team's speed to market, focusing on how well the team is able to balance innovation with the practical demands of delivering solutions in a timely manner. Ideally a team is well-balanced, meaning that they have approximately equal ratios members in each of the score quadrants."
            )
        
        elif focus == "Strategic Agility Index":
            prompt = (
                f"Generate a detailed description for the team '{project_name}' with focus on the Managing Complexity dimension.\n\n"
                f"Here are the aggregate measures of the team scores per dimension:\n"
                f"{'\n'.join(aggregate_texts)}\n\n"
                f"Provide an analysis that describes how the Managing Complexity score influences the team's ability to manage both complex and complicated problems. Consider how the score affects the team's adaptability, decision-making, and overall strategic execution. Ideally a team is well-balanced, meaning that they have approximately equal ratios members in each of the score quadrants."
            )
        
        elif focus == "Strategic Hire Analysis":
            prompt = (
                f"Generate a detailed description for the team '{project_name}' with focus on the five result areas created by the Attachment and Exploration dimensions.\n\n"
                f"Here are the aggregate measures of the team scores per dimension:\n"
                f"{'\n'.join(aggregate_texts)}\n\n"
                f"Provide an analysis that uses the provided Attachment and Exploration scores to categorize individuals into one of the five result areas from the Strategic Hire Analysis: 1. Relationship-Optimization Quadrant (High Attachment score combined with Low Exploration Score), 2.Content-Optimization Quadrant (Low Attachment score with Low Exploration Score) , 3. Relationship-Exploration Quadrant (High Attachment score combined with High Exploration Score), 4. Content-Exploration Quadrant (Low Attachment Score combined with High Exploration score), 5. Strategic Execution Zone (Medium Attachment score combined with medium Exploration score). Based on the categorization, describe the team’s strengths and  potential gaps in performance. Be sure to reflect on how the combination of the Attachment and Exploration dimensions shapes each individual’s contribution to the team. Ideally, the team has a nice spread across the 5 different result-areas. When there is an imbalance in these quadrants (e.g., over or under-representation in one area), suggest potential consequences or blind spots, and provide recommendations for balancing the team."
            )
        
        elif focus == "Business Performance":
            prompt = (
                f"Generate a detailed description for the team '{project_name}' with focus on the Business Performance Dialogue, based on Attachment and Exploration scores.\n\n"
                f"Here are the aggregate measures of the team scores per dimension:\n"
                f"{'\n'.join(aggregate_texts)}\n\n"
                f"Provide an analysis that identifies where team members fall within each of the three phases of the business perfomance dialogue based on their score profiles: 1. Changing with people (High Exploration combined with high Attachment), 2. Operational Core (Mid-range exploration combined with mid-range attachment), 3. Structured delivery (low exploration combined with low attachment). Highlights how these contributions shape the team's performance and collaboration in each phase (e.g., driving change, converting ideas into projects, ensuring execution). Points out any gaps or underrepresented contributions in specific phases, particularly where contributions from either Attachment or Exploration might be lacking. Suggests strategies to address these gaps, ensuring smoother transitions across phases. Ideally, a team has approximately equal ratios of teammembers in each of the three phases."
            )
        
        elif focus == "Safeguarding Innovation":
            prompt = (
                f"Generate a detailed description for the team '{project_name}' with focus on Safeguarding Innovation, based on the Exploration and Attachment scores.\n\n"
                f"Here are the aggregate measures of the team scores per dimension:\n"
                f"{'\n'.join(aggregate_texts)}\n\n"
                f"Provide an analysis that identifies where team members fall within each of the three phases: 1. Inventive Exploration (High Exploration combined Low Attachment), 2. Operational Testing (Mid-range exploration combined with mid-range attachment), 3. Sustaining with people (low exploration combined with high attachment).  Highlights any gaps or areas where the team may lack strong contributions.. Suggests how the team can better navigate the transition of ideas from research and experimentation to practical, sustainable implementation. Proposes strategies to address these gaps, including possible adjustments to team roles or introducing new perspectives to ensure successful embedding of innovations."
            )
   
    # Request from OpenAI API using client
        response = client.chat.completions.create(
            model=os.getenv("MODEL_ID"),  # Replace with your fine-tuned model ID
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

    # Add team member scores to the response
    for _, row in project_data.iterrows():
        member_info = {
            "name": row["Participant"],
            "attachment_score": row["attachment score"],
            "exploration_score": row["exploration score"],
            "managing_complexity_score": row["managing complexity score"]
        }
        team_member_scores.append(member_info)

    # Return descriptions and team member scores
    return {"descriptions": descriptions, "team_member_scores": team_member_scores}

