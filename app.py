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
import markdown
from pydub import AudioSegment
import uuid

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploaded_images'


# Create upload folder if it doesn't exist
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

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
    result = markdown.markdown(response)
    # print(result)
    return render_template('index.html',rel=result)

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



@app.route('/upload', methods=['POST'])
def upload_image():
    if 'image' not in request.files:
        return "No file part"
    
    file = request.files['image']
    
    if file.filename == '':
        return "No selected file"

    if file:
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        file.save(file_path)
        print(file_path)

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
        food_label = classify_food_image(file_path, classifier_model, class_names)
        clean_label = food_label.replace('_', ' ').lower()

        user_input = f"This is a photo of {clean_label}. Based on this food item, suggest a healthy meal plan or a healthier alternative if needed."
        print(f"\n--- Classified Food from Image ---\n{user_input}")
    prompt = build_prompt(user_input)
    response = get_groq_response(prompt)

    print("\n--- AI Meal Plan and Explanation ---\n")
    print(response)
    result = markdown.markdown(response)
    return render_template('index.html',rel=result)

UPLOAD_FOLDER = 'static/uploaded_audio'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

#Audio Handling
@app.route('/upload-audio', methods=['POST'])
def upload_audio():
    audio = request.files.get('audio')
    if not audio:
        return jsonify({"error": "No audio received"}), 400

    # Save the original file temporarily with a unique name
    original_ext = os.path.splitext(audio.filename)[1]  # e.g. '.webm'
    temp_filename = f"temp_{uuid.uuid4().hex}{original_ext}"
    temp_filepath = os.path.join(UPLOAD_FOLDER, temp_filename)
    audio.save(temp_filepath)

    try:
        # Convert to .wav using pydub
        audio_segment = AudioSegment.from_file(temp_filepath)
        wav_filename = f"{uuid.uuid4().hex}.wav"
        wav_filepath = os.path.join(UPLOAD_FOLDER, wav_filename)
        
        # Export to WAV format
        audio_segment.export(wav_filepath, format="wav")

        # Delete the original temp file (the webm file)
        os.remove(temp_filepath)

        print(wav_filepath)
        recognizer = sr.Recognizer()
        with sr.AudioFile(wav_filepath) as source:
            audio = recognizer.record(source)
        try:
            audio_output = recognizer.recognize_google(audio)
        except sr.UnknownValueError:
            return "Sorry, could not understand the audio."
        
        prompt = build_prompt(audio_output)
        response = get_groq_response(prompt)

        # print("\n--- AI Meal Plan and Explanation ---\n")
        print(response)

        # return jsonify({
        #     "message": response,
        #     "wav_file": wav_filename,
        #     "wav_path": wav_filepath
        # })
    
        result = markdown.markdown(response)
        rel = result  # whatever your processing returns
        return jsonify({"rel": rel})
    except Exception as e:
        return jsonify({"error": f"Conversion failed: {str(e)}"}), 500
    
    # return render_template('index.html',rel=result)

   

if __name__ == '__main__':
    app.run(debug=True)





