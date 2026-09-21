from flask import Flask, render_template, request, jsonify
from flask_cors import CORS
import json
import os
import random
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

app = Flask(__name__)
CORS(app)

# groq uses openai's sdk since their api is openai-compatible
client = OpenAI(
    api_key=os.getenv("OPENAI_API_KEY"),
    base_url=os.getenv("GROQ_BASE_URL", "https://api.groq.com/openai/v1")
)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "skin-model")

# TODO: switch this off once real model weights are available
# (LFS pull isn't working rn, so just reading labels for now)
STUB_MODE = True

with open(os.path.join(MODEL_PATH, "config.json")) as f:
    model_config = json.load(f)

CLASS_LABELS = list(model_config["id2label"].values())

print(f"loaded {len(CLASS_LABELS)} labels, stub mode = {STUB_MODE}")


@app.route('/')
def home():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files and 'image' not in request.files:
        return jsonify({'error': 'No file uploaded'})

    file = request.files.get('file') or request.files.get('image')

    age = request.form.get('age', 'Not specified')
    gender = request.form.get('gender', 'Not specified')
    history = request.form.get('history', 'No prior history mentioned')

    if file.filename == '':
        return jsonify({'error': 'No file selected'})

    try:
        if STUB_MODE:
            # fake it till the real weights show up
            top_3_labels = random.sample(CLASS_LABELS, 3)
            top_3 = []
            remaining = 100.0
            for i, label in enumerate(top_3_labels):
                conf = round(random.uniform(10, remaining), 2) if i < 2 else round(remaining, 2)
                remaining -= conf
                top_3.append({'class': label, 'confidence': conf})
            top_3.sort(key=lambda x: x['confidence'], reverse=True)

            predicted_class = top_3[0]['class']
            confidence = top_3[0]['confidence']
        else:
            # real model path goes here later
            pass

        # ask groq for the detailed writeup
        detailed_analysis = {
            "overview": "Analysis could not be generated at this time.",
            "causes": ["Wait for AI"],
            "routine": ["Consult a dermatologist"],
            "whenToSee": "Consult a professional.",
            "tips": ["Stay healthy"]
        }
        try:
            prompt = f"""
            As an AI medical assistant, provide a professional but easy-to-understand detailed analysis for the following skin condition prediction:
            - Prediction: {predicted_class}
            - Confidence: {confidence:.2f}%

            Patient Information:
            - Age: {age}
            - Gender: {gender}
            - Medical History: {history}

            Return the response ONLY as a JSON object with the following keys:
            - "overview": A professional description of the condition (2-3 sentences).
            - "causes": An array of 3-4 common causes.
            - "routine": An array of 4-5 recommended skincare steps.
            - "whenToSee": Advice on when to consult a doctor.
            - "decision": A detailed 3-4 sentence final professional recommendation, explaining the reasoning and immediate next steps.
            - "tips": 2-3 general lifestyle tips.

            Keep the content professional and concise.
            """

            response = client.chat.completions.create(
                #model="llama-3.3-70b-versatile",
                model="openai/gpt-oss-120b",
                messages=[
                    {"role": "system", "content": "You are a helpful medical assistant specializing in dermatology. You respond only in JSON."},
                    {"role": "user", "content": prompt}
                ],
                response_format={"type": "json_object"}
            )
            detailed_analysis = json.loads(response.choices[0].message.content)
        except Exception as ai_err:
            print(f"groq error: {ai_err}")

        return jsonify({
            'prediction': predicted_class,
            'confidence': confidence,
            'top_3': top_3,
            'detailed_analysis': detailed_analysis
        })

    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5001)