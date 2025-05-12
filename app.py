import os
from groq import Groq

# Set your API key
groq_api_key = "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"  # or directly paste it
client = Groq(api_key=groq_api_key)

# Input: user goals and preferences
user_input = {
    "health_goal": "weight loss",
    "diet": "vegetarian",
    "allergies": ["nuts"],
    "activity_level": "moderate",
    "meal_type": "dinner"
}

# Prompt construction for LLM
prompt = f"""
You are a smart AI Nutrition Assistant.

Create a personalized {user_input['meal_type']} plan for someone with the following profile:
- Health goal: {user_input['health_goal']}
- Diet: {user_input['diet']}
- Allergies: {', '.join(user_input['allergies'])}
- Activity level: {user_input['activity_level']}

Then explain briefly why the chosen meal is good for their goal.
"""

# Call Groq (use Mixtral or LLaMA3 for better reasoning)
response = client.chat.completions.create(
    model="llama3-8b-8192",  # or "llama3-8b-8192"
    messages=[{"role": "user", "content": prompt}],
    temperature=0.7,
)

# Output response
answer = response.choices[0].message.content
print("\n--- AI Meal Plan and Explanation ---\n")
print(answer)

# if __name__ == "__main__":
#     # run()

