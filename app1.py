import os
from groq import Groq
import pytesseract
from PIL import Image
import speech_recognition as sr
import tempfile
import easyocr
import torch
import torch.nn as nn
from torchvision import transforms, models
from torchvision import datasets
# pytesseract.pytesseract.tesseract_cmd = r"C:\\Users\\rohid\\OneDrive\\Desktop\\The Smartest AI Nutrition Assistant\\myenv\\Lib\\site-packages\\pytesseract"

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

# def handle_image_input(image_path):
#     image = Image.open(image_path)
#     extracted_text = pytesseract.image_to_string(image)
#     return extracted_text.strip()

def handle_image_input(image_path):
    reader = easyocr.Reader(['en'], gpu=False)
    result = reader.readtext(image_path, detail=0)
    extracted_text = ' '.join(result)
    return extracted_text.strip()

def handle_audio_input(audio_path):
    recognizer = sr.Recognizer()
    with sr.AudioFile(audio_path) as source:
        audio = recognizer.record(source)
    try:
        return recognizer.recognize_google(audio)
    except sr.UnknownValueError:
        return "Sorry, could not understand the audio."
    
# === FOOD IMAGE CLASSIFIER ===

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_model(model_path, class_names):
    model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    model.fc = nn.Linear(model.fc.in_features, len(class_names))
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    return model.to(device)

def classify_food_image(image_path, model, class_names):
    image = Image.open(image_path).convert("RGB")
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])
    input_tensor = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = model(input_tensor)
        _, predicted = torch.max(outputs, 1)
    return class_names[predicted.item()]

    



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
        # Load class names from folder structure
        temp_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor()
        ])
        temp_dataset = datasets.ImageFolder("./food_data/train", transform=temp_transform)
        class_names = temp_dataset.classes

        # Load model and classify image
        model_path = 'food_classifier.pth'
        classifier_model = load_model(model_path, class_names)
        food_label = classify_food_image(input_path, classifier_model, class_names)
        clean_label = food_label.replace('_', ' ').lower()

        user_input = f"This is a photo of {clean_label}. Based on this food item, suggest a healthy meal plan or a healthier alternative if needed."
        print(f"\n--- Classified Food from Image ---\n{user_input}")

    elif input_type == "audio":
        user_input = handle_audio_input(input_path)

    else:
        print("Invalid input type. Choose from: text, image, audio, video.")
        return

    prompt = build_prompt(user_input)
    response = get_groq_response(prompt)

    print("\n--- AI Meal Plan and Explanation ---\n")
    print(response)


# def main(input_type, input_path=None):
#     if input_type == "text":
#         user_input = handle_text_input()

#     elif input_type == "image":
#         # Load class names from folder structure
#         temp_transform = transforms.Compose([
#             transforms.Resize((224, 224)),
#             transforms.ToTensor()
#         ])
#         temp_dataset = datasets.ImageFolder("./food_data/train", transform=temp_transform)
#         class_names = temp_dataset.classes

#         # Load model and classify image
#         model_path = 'food_classifier.pth'
#         classifier_model = load_model(model_path, class_names)
#         food_label = classify_food_image(input_path, classifier_model, class_names)

#         user_input = f"The user provided an image of {food_label.replace('_', ' ')}."
#         print(f"\n--- Classified Food from Image ---\n{user_input}")

#     elif input_type == "audio":
#         user_input = handle_audio_input(input_path)

#     else:
#         print("Invalid input type. Choose from: text, image, audio, video.")
#         return

#     prompt = build_prompt(user_input)
#     response = get_groq_response(prompt)

#     print("\n--- AI Meal Plan and Explanation ---\n")
#     print(response)


# def main(input_type, input_path=None):
#     if input_type == "text":
#         user_input = handle_text_input()
#     elif input_type == "image":
#         user_input = handle_image_input(input_path)
#     elif input_type == "audio":
#         user_input = handle_audio_input(input_path)
#     else:
#         print("Invalid input type. Choose from: text, image, audio, video.")
#         return

#     prompt = build_prompt(user_input)
#     response = get_groq_response(prompt)

#     print("\n--- AI Meal Plan and Explanation ---\n")
#     print(response)

# === USAGE ===
# Example usage:
# main("text")
# main("image", "label.jpg")
# main("audio", "input.wav")
# main("video", "diet_tips.mp4")

if __name__ == "__main__":
    # Change input type and file path as needed
    # main("text")
    main("image", "eggs.jpg")
    # main("audio", "recorded_input.wav")
    # main("video", "health_goals.mp4")
