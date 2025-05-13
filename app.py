from flask import *
import os
from groq import Groq
import pytesseract
from PIL import Image
import speech_recognition as sr
# import tempfile
import easyocr
import torch
import torch.nn as nn
from torchvision import transforms, models
from torchvision import datasets

app = Flask(__name__)

# === SET GROQ API KEY ===
groq_api_key = "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"
client = Groq(api_key=groq_api_key)

@app.route('/')
def Index():
    return render_template('index.html')

def build_prompt(user_input_text_or_dict):
    if isinstance(user_input_text_or_dict, dict):
        return f"""
You are a smart AI Nutrition Assistant.

Create a personalized {user_input_text_or_dict['meal_type']} plan for someone with the following profile:
- Health goal: {user_input_text_or_dict['health_goal']}
- Diet: {user_input_text_or_dict['diet']}
- Allergies: {', '.join(user_input_text_or_dict['allergies'])}
- Activity level: {user_input_text_or_dict['activity_level']}

Then explain briefly why the chosen meal is good for their goal.
"""
    else:
        return f"""
You are a smart AI Nutrition Assistant.

The user provided the following input:
\"{user_input_text_or_dict.strip()}\"

Extract goals, preferences, and generate a suitable meal plan and explanation.
"""

def get_groq_response(prompt):
    response = client.chat.completions.create(
        model="llama3-8b-8192",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.7,
    )
    return response.choices[0].message.content.strip()

@app.route('/',methods=['POST'])
def handle_text_input():
    health_goal = request.form.get('hg')
    diet = request.form.get('diet')
    allergies = request.form.get('allergies')
    activity_level = request.form.get('al')
    meal_type = request.form.get('mt')
    print(health_goal,diet,allergies,activity_level,meal_type)

    user_input = {
        "health_goal": health_goal,
        "diet": diet,
        "allergies": allergies,
        "activity_level": activity_level,
        "meal_type": meal_type
    }
    prompt = build_prompt(user_input)
    response = get_groq_response(prompt)

    print("\n--- AI Meal Plan and Explanation ---\n")
    result = response
    return render_template('index.html',rel=result)



if __name__ == '__main__':
    app.run(debug=True)





