import os
from groq import Groq
import pytesseract
from PIL import Image
import speech_recognition as sr
import tempfile
pytesseract.pytesseract.tesseract_cmd = r"C://Users//rohid//OneDrive//Desktop//The Smartest AI Nutrition Assistant//myenv//Lib//site-packages//pytesseract"

# === SET YOUR GROQ API KEY ===
groq_api_key = "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"
client = Groq(api_key=groq_api_key)

# === INPUT HANDLERS ===

def handle_text_input():
    return {
        "health_goal": "weight loss",
        "diet": "vegetarian",
        "allergies": ["nuts"],
        "activity_level": "moderate",
        "meal_type": "dinner"
    }

def handle_image_input(image_path):
    image = Image.open(image_path)
    extracted_text = pytesseract.image_to_string(image)
    return extracted_text.strip()

def handle_audio_input(audio_path):
    recognizer = sr.Recognizer()
    with sr.AudioFile(audio_path) as source:
        audio = recognizer.record(source)
    try:
        return recognizer.recognize_google(audio)
    except sr.UnknownValueError:
        return "Sorry, could not understand the audio."


# === LLM PROMPT BUILDER ===

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

# === LLM CALL ===

def get_groq_response(prompt):
    response = client.chat.completions.create(
        model="llama3-8b-8192",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.7,
    )
    return response.choices[0].message.content.strip()

# === MAIN ===

def main(input_type, input_path=None):
    if input_type == "text":
        user_input = handle_text_input()
    elif input_type == "image":
        user_input = handle_image_input(input_path)
    elif input_type == "audio":
        user_input = handle_audio_input(input_path)
    else:
        print("Invalid input type. Choose from: text, image, audio, video.")
        return

    prompt = build_prompt(user_input)
    response = get_groq_response(prompt)

    print("\n--- AI Meal Plan and Explanation ---\n")
    print(response)

# === USAGE ===
# Example usage:
# main("text")
# main("image", "label.jpg")
# main("audio", "input.wav")
# main("video", "diet_tips.mp4")

if __name__ == "__main__":
    # Change input type and file path as needed
    # main("text")
    main("image", "pizza.jpg")
    # main("audio", "recorded_input.wav")
    # main("video", "health_goals.mp4")
