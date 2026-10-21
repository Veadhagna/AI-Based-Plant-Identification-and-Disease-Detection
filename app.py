
from flask import Flask, render_template, request, jsonify
from deep_translator import GoogleTranslator
import tensorflow as tf
import numpy as np
import json
import os
from dotenv import load_dotenv
from groq import Groq
from werkzeug.utils import secure_filename

app = Flask(__name__)

# ============================================================
# ENVIRONMENT AND GROQ CONFIGURATION
# ============================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
load_dotenv(os.path.join(BASE_DIR, ".env"))

groq_api_key = os.getenv("GROQ_API_KEY")
groq_client = Groq(api_key=groq_api_key) if groq_api_key else None

# ============================================================
# LOAD MODELS
# ============================================================

plant_model_path = os.path.join(
    BASE_DIR, "ml_v2", "results",
    "plant_mobilenetv2", "best_plant_mobilenetv2.keras"
)
plant_model = tf.keras.models.load_model(plant_model_path)

disease_model_path = os.path.join(
    BASE_DIR, "ml_v2", "results",
    "disease_model", "best_disease_model.keras"
)
disease_model = tf.keras.models.load_model(disease_model_path)

with open(
    os.path.join(
        BASE_DIR, "ml_v2", "results",
        "disease_model", "class_names.json"
    ),
    "r",
    encoding="utf-8"
) as f:
    disease_classes = json.load(f)

if isinstance(disease_classes, dict):
    if "class_names" in disease_classes:
        disease_classes = disease_classes["class_names"]
    else:
        disease_classes = list(disease_classes.values())

print(f"Disease model loaded: {len(disease_classes)} classes")

# ============================================================
# DISEASE INFORMATION
# ============================================================

plant_diseases = {
    "Aloevera": {
        "status": "Check for Leaf Spots",
        "details": "Brown spots on leaves may indicate fungal infection",
        "solution": "Remove affected leaves and improve air circulation"
    },
    "Tulasi": {
        "status": "Watch for Root Rot",
        "details": "Yellowing leaves and wilting may indicate root problems",
        "solution": "Reduce watering frequency and ensure good drainage"
    },
    "Neem": {
        "status": "Monitor for Powdery Mildew",
        "details": "White powdery coating on leaves",
        "solution": "Apply neem oil solution and improve air circulation"
    },
    "Mint": {
        "status": "Check for Rust",
        "details": "Orange-brown spots on leaves",
        "solution": "Remove affected leaves and avoid overhead watering"
    },
    "Curry Leaf": {
        "status": "Watch for Scale Insects",
        "details": "Small brown bumps on stems and leaves",
        "solution": "Apply insecticidal soap or neem oil"
    },
    "Hibiscus": {
        "status": "Monitor for Aphids",
        "details": "Curling leaves and sticky residue",
        "solution": "Spray with strong water stream or use insecticidal soap"
    },
    "Rose": {
        "status": "Check for Black Spot",
        "details": "Black spots with yellow halos on leaves",
        "solution": "Remove affected leaves and apply fungicide"
    },
    "Jasmine": {
        "status": "Watch for Spider Mites",
        "details": "Fine webbing and stippled leaves",
        "solution": "Increase humidity and apply miticide if needed"
    },
    "Lemon": {
        "status": "Monitor for Citrus Canker",
        "details": "Raised corky lesions on leaves and fruit",
        "solution": "Remove affected parts and use copper-based fungicide"
    },
    "Mango": {
        "status": "Check for Anthracnose",
        "details": "Dark spots on leaves and fruit",
        "solution": "Apply fungicide and improve air circulation"
    }
}

# ============================================================
# MEDICINAL USES
# ============================================================

medicinal_uses = {
    "Aloevera": "(Aloe vera) - Used for skin hydration, healing burns, and treating digestive issues.",
    "Amla": "(Phyllanthus emblica) - Rich in Vitamin C, it boosts immunity and aids digestion.",
    "Amruta Balli": "(Tinospora cordifolia) - Known for boosting immunity and treating fever and diabetes.",
    "Arali": "(Nerium oleander) - Though toxic, it has been used in traditional medicine for heart disease and skin conditions (use with caution).",
    "Ashoka": "(Saraca asoca) - Used in Ayurvedic medicine to treat menstrual disorders and improve skin health.",
    "Ashwagandha": "(Withania somnifera) - Reduces stress, anxiety, and improves vitality.",
    "Avacado": "(Persea americana) - Rich in healthy fats, it's good for heart health and skin.",
    "Bamboo": "(Bambusa spp.) - Used in traditional medicine to treat infections and aid in bone health.",
    "Basale": "(Basella alba) - Helps in treating anemia and boosting immunity due to its high iron content.",
    "Betel": "(Piper betle) - Chewed to improve digestion and treat bad breath.",
    "Betel Nut": "(Areca catechu) - Used as a stimulant and to treat digestive issues.",
    "Brahmi": "(Bacopa monnieri) - Enhances memory and reduces anxiety.",
    "Castor": "(Ricinus communis) - Used as a laxative and to treat skin conditions.",
    "Curry Leaf": "(Murraya koenigii) - Improves digestion and manages diabetes.",
    "Doddapatre": "(Plectranthus amboinicus) - Treats coughs, colds, and skin infections.",
    "Ekka": "(Calotropis gigantea) - Used for treating skin diseases and joint pain.",
    "Ganike": "(Clerodendrum serratum) - Traditionally used to treat respiratory conditions.",
    "Guava": "(Psidium guajava) - Rich in Vitamin C, it boosts immunity and aids digestion.",
    "Geranium": "(Pelargonium graveolens) - Used for its anti-inflammatory and astringent properties in skincare.",
    "Henna": "(Lawsonia inermis) - Used for its cooling properties and to treat skin conditions.",
    "Hibiscus": "(Hibiscus rosa-sinensis) - Used to lower blood pressure and treat hair loss.",
    "Honge": "(Pongamia pinnata) - Used for its anti-inflammatory properties and to treat skin disorders.",
    "Insulin": "(Costus igneus) - Known to help regulate blood sugar levels.",
    "Jasmine": "(Jasminum spp.) - Used in aromatherapy to reduce stress and improve skin health.",
    "Lemon": "(Citrus limon) - Boosts immunity and aids digestion due to its high Vitamin C content.",
    "Lemon Grass": "(Cymbopogon citratus) - Used to relieve pain and inflammation and aid digestion.",
    "Mango": "(Mangifera indica) - Rich in antioxidants, it's beneficial for the skin and digestion.",
    "Mint": "(Mentha spp.) - Relieves digestive issues and freshens breath.",
    "Nagadali": "(Rauvolfia serpentina) - Used to treat high blood pressure and mental disorders.",
    "Neem": "(Azadirachta indica) - Known for its antibacterial properties and used to treat skin conditions.",
    "Nithyapushpa": "(Vinca rosea) - Used in treating diabetes and cancer (under medical supervision).",
    "Nooni": "(Morinda citrifolia) - Known for its immune-boosting and pain-relieving properties.",
    "Papaya": "(Carica papaya) - Aids digestion and improves skin health.",
    "Pepper": "(Piper nigrum) - Used to improve digestion and as an anti-inflammatory.",
    "Pomegranate": "(Punica granatum) - Rich in antioxidants, it promotes heart health and reduces inflammation.",
    "Raktachandini": "(Pterocarpus santalinus) - Used to treat skin disorders and improve blood circulation.",
    "Rose": "(Rosa spp.) - Used in skincare for its anti-inflammatory and antioxidant properties.",
    "Sapota": "(Manilkara zapota) - Boosts energy and aids in digestion.",
    "Tulasi": "(Ocimum sanctum) - Known for its immune-boosting and anti-inflammatory properties.",
    "Wood Sorel": "(Oxalis corniculata) - Used to treat fever, inflammation, and digestive issues."
}

# ============================================================
# FILE VALIDATION
# ============================================================

ALLOWED_EXTENSIONS = {"jpg", "jpeg", "png"}


def allowed_file(filename):
    return (
        "." in filename
        and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS
    )


# ============================================================
# PLANT AND DISEASE PREDICTION
# ============================================================

def get_plant_info(image_path, selected_plant=None):
    plant_classes = [
        "Aloevera", "Amla", "Amruta_Balli", "Arali", "Ashoka",
        "Ashwagandha", "Avacado", "Bamboo", "Basale", "Betel",
        "Betel_Nut", "Brahmi", "Castor", "Curry_Leaf", "Doddapatre",
        "Ekka", "Ganike", "Gauva", "Geranium", "Henna", "Hibiscus",
        "Honge", "Insulin", "Jasmine", "Lemon", "Lemon_grass",
        "Mango", "Mint", "Nagadali", "Neem", "Nithyapushpa",
        "Nooni", "Pappaya", "Pepper", "Pomegranate", "Raktachandini",
        "Rose", "Sapota", "Tulasi", "Wood_sorel"
    ]

    supported_plants = {
        "Aloevera", "Basale", "Betel", "Castor", "Gauva",
        "Jasmine", "Lemon", "Mango", "Neem", "Pappaya", "Rose"
    }

    img = tf.keras.utils.load_img(image_path, target_size=(224, 224))
    img_array = tf.keras.utils.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = tf.keras.applications.mobilenet_v2.preprocess_input(img_array)

    plant_prediction = plant_model.predict(img_array, verbose=0)
    plant_index = int(np.argmax(plant_prediction[0]))
    plant_confidence = float(np.max(plant_prediction[0]))
    detected_plant = plant_classes[plant_index]


    # Reject low-confidence plant predictions.
# This is a provisional threshold and must be tested.
    if plant_confidence < 0.60:
        return {
            "plant": "Uncertain — please upload a clear plant leaf image",
            "detected_plant": "Uncertain",
            "plant_confidence": round(plant_confidence * 100, 2),
            "uses": "Plant information unavailable because identification was uncertain.",
            "disease": "Not available",
            "disease_confidence": 0.0,
            "disease_status": "Prediction skipped",
            "disease_details": "The image did not meet the minimum plant-identification confidence threshold.",
            "disease_solution": "Upload a clear, well-lit image showing a plant leaf."
    }

    plant_name = selected_plant if selected_plant else detected_plant

    medicinal_key_map = {
        "Pappaya": "Papaya",
        "Gauva": "Guava",
        "Amruta_Balli": "Amruta Balli",
        "Betel_Nut": "Betel Nut",
        "Curry_Leaf": "Curry Leaf",
        "Lemon_grass": "Lemon Grass",
        "Wood_sorel": "Wood Sorel"
    }

    medicinal_key = medicinal_key_map.get(plant_name, plant_name)
    uses = medicinal_uses.get(
        medicinal_key, "No medicinal use information available."
    )

    disease_name = "Not available"
    disease_confidence = 0.0
    disease_status = "Disease detection unavailable"
    disease_details = (
        "Detailed disease/condition detection is currently "
        "available for 11 supported plants."
    )
    disease_solution = (
        "Please select one of the supported plants for "
        "plant-specific disease/condition detection."
    )

    if plant_name in supported_plants:
        disease_img = tf.keras.utils.load_img(
            image_path, target_size=(224, 224)
        )
        disease_array = tf.keras.utils.img_to_array(disease_img)
        disease_array = np.expand_dims(disease_array, axis=0)

        disease_prediction = disease_model.predict(disease_array, verbose=0)
        expected_prefix = plant_name + "_"

        matching_indices = [
            i for i, name in enumerate(disease_classes)
            if name.startswith(expected_prefix)
        ]

        if matching_indices:
            matching_scores = disease_prediction[0][matching_indices]
            best_position = int(np.argmax(matching_scores))
            disease_index = matching_indices[best_position]
            raw_disease = disease_classes[disease_index]
            disease_confidence = float(disease_prediction[0][disease_index])

    # Avoid presenting a low-confidence result as reliable.
            if disease_confidence < 0.60:
                disease_name = raw_disease[len(expected_prefix):]
                disease_status = "Tentative prediction — low confidence"
                disease_details = (
                    f"The model's highest-scoring condition for {plant_name} "
                    f"is {disease_name}, but confidence is low. "

                )
                disease_solution = get_disease_care(disease_name)

            else:
                disease_name = raw_disease[len(expected_prefix):]
                disease_status = disease_name
                disease_details = (
                    f"The model's highest-scoring condition for "
                    f"{plant_name} is {disease_name}."
                )
                disease_solution = get_disease_care(disease_name)

    else:
        disease_status = "Disease detection unavailable"
        disease_details = f"No disease classes were found for {plant_name}."
        disease_solution = (
            "Please verify the selected plant and the disease "
            "model's supported classes."
        )

    return {
        "plant": plant_name,
        "detected_plant": detected_plant,
        "plant_confidence": round(plant_confidence * 100, 2),
        "uses": uses,
        "disease": disease_name,
        "disease_confidence": round(disease_confidence * 100, 2),
        "disease_status": disease_status,
        "disease_details": disease_details,
        "disease_solution": disease_solution
    }


def get_disease_care(disease_name):
    """Return general care guidance for the predicted plant condition."""
    disease = disease_name.lower()

    # Healthy leaves
    if "healthy" in disease:
        return (
            "The model classified this leaf as healthy. Maintain suitable "
            "watering and sunlight, inspect leaves regularly, and monitor "
            "for any new symptoms."
        )

    # Pest-related damage
    if any(word in disease for word in [
        "pest", "insect", "miner", "caterpillar", "mite", "weevil", "midge"
    ]):
        return (
            "Inspect both sides of the leaves for insects, eggs, and feeding "
            "damage. Check nearby plants and identify the pest before choosing "
            "a control method. Seek local agricultural advice if damage spreads."
        )

    # Fungal or mould-related conditions
    if any(word in disease for word in [
        "fung", "powdery_mildew", "mould", "mold", "anthracnose",
        "alternaria", "rust", "downy_mildew", "leaf_spot",
        "leaf_blight", "scorch", "rot"
    ]):
        return (
            "Remove severely affected plant material where appropriate, avoid "
            "unnecessary wetting of leaves, and improve airflow around the "
            "plant. Confirm the cause before using any fungicide; seek local "
            "agricultural advice if symptoms continue to spread."
        )

    # Bacterial conditions
    if any(word in disease for word in [
        "bacterial", "bacterial_spot", "canker"
    ]):
        return (
            "Avoid handling wet plants, clean tools between plants, and remove "
            "severely affected material where appropriate. Confirm whether the "
            "condition is bacterial before treatment, and seek agricultural "
            "advice if symptoms spread."
        )

    # Viral conditions
    if any(word in disease for word in [
        "virus", "curl", "ring_spot", "yld"
    ]):
        return (
            "Check for leaf curling, unusual patterns, and possible insect "
            "vectors. Isolate affected plants when practical and seek local "
            "agricultural advice to confirm the cause before deciding on control."
        )

    # Other or unclassified conditions
    return (
        "Inspect the leaves for changes in colour, spots, holes, or wilting. "
        "Compare symptoms with reliable plant-health guidance and consult an "
        "agricultural expert before applying treatment."
    )

# ============================================================
# WEB ROUTES
# ============================================================

@app.route("/")
def home():
    return render_template("home.html")


@app.route("/index")
def index():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    if "image" not in request.files:
        return jsonify({"error": "No file part"}), 400

    file = request.files["image"]

    if file.filename == "":
        return jsonify({"error": "No selected file"}), 400

    if not allowed_file(file.filename):
        return jsonify({"error": "Invalid file format"}), 400

    filename = secure_filename(file.filename)
    if not filename:
        return jsonify({"error": "Invalid filename"}), 400

    upload_dir = os.path.join(BASE_DIR, "uploads")
    os.makedirs(upload_dir, exist_ok=True)
    file_path = os.path.join(upload_dir, filename)

    try:
        file.save(file_path)
        selected_plant = request.form.get("plant")
        plant_info = get_plant_info(file_path, selected_plant)

        return jsonify({
            "plant": plant_info["plant"],
            "detected_plant": plant_info["detected_plant"],
            "plant_confidence": plant_info["plant_confidence"],
            "uses": plant_info["uses"],
            "disease": plant_info["disease"],
            "disease_confidence": plant_info["disease_confidence"],
            "disease_status": plant_info["disease_status"],
            "disease_details": plant_info["disease_details"],
            "disease_solution": plant_info["disease_solution"]
        })
    except Exception:
        app.logger.exception("Plant prediction failed")
        return jsonify({"error": "Plant prediction failed. Check the server logs."}), 500


# ============================================================
# CHATBOT LANGUAGE HELPERS
# ============================================================

def detect_supported_language(text):
    """Detect Kannada or Hindi script; otherwise default to English."""
    if any("\u0C80" <= ch <= "\u0CFF" for ch in text):
        return "kn"
    if any("\u0900" <= ch <= "\u097F" for ch in text):
        return "hi"
    return "en"




def translate_text(text, source, target):
    """Translate text using Groq for English, Kannada, and Hindi."""
    if not isinstance(text, str) or not text.strip():
        return text

    if source == target:
        return text

    language_names = {
        "en": "English",
        "kn": "Kannada",
        "hi": "Hindi",
    }

    source_name = language_names.get(source)
    target_name = language_names.get(target)

    if not source_name or not target_name:
        raise ValueError(
            f"Unsupported translation: {source} -> {target}"
        )

    if groq_client is None:
        raise RuntimeError("Groq API key is not configured.")

    completion = groq_client.chat.completions.create(
        model="openai/gpt-oss-20b",
        messages=[
            {
                "role": "system",
                "content": (
                    f"Translate the text from {source_name} to {target_name}. "
                    "Return only the translation. Preserve plant names "
                    "and scientific names. Do not answer the text."
                ),
            },
            {
                "role": "user",
                "content": text.strip(),
            },
        ],
        temperature=0,
    )

    translated = completion.choices[0].message.content

    if not isinstance(translated, str) or not translated.strip():
        raise ValueError("Groq returned an empty translation.")

    return translated.strip()


# ============================================================
# CHATBOT
# ============================================================


@app.route("/chat", methods=["GET", "POST"])
def chat():

    if request.method == "GET":
        user_msg = request.args.get("msg", "")
        selected_language = request.args.get("language", "auto")
    else:
        request_data = request.get_json(silent=True) or {}
        user_msg = request_data.get("msg", "")
        selected_language = request_data.get("language", "auto")

    if not isinstance(user_msg, str) or not user_msg.strip():
        return jsonify({
            "response": "Please enter a question about a medicinal plant.",
            "language": "en"
        }), 400

    user_msg = user_msg.strip()

# Use the selected language when provided; otherwise detect it.
    if selected_language in ("en", "kn", "hi"):
        src_lang = selected_language
    else:
        src_lang = detect_supported_language(user_msg)

    original_question = user_msg.lower().strip()



    # Translate the question into English when possible.
    try:
        translated_input = translate_text(user_msg, src_lang, "en")
        if not isinstance(translated_input, str) or not translated_input.strip():
            translated_input = user_msg
    except Exception:
        app.logger.exception("Input translation failed")
        translated_input = user_msg

    question = translated_input.lower().strip()

    # Search both the original question and translated question.
    # This helps when a plant name remains in Kannada or Hindi.
    search_text = f"{question} {original_question}"

    aliases = {
        "aloe vera": "Aloevera",
        "avocado": "Avacado",
        "guava": "Guava",
        "papaya": "Papaya",
        "holy basil": "Tulasi",
        "tulsi": "Tulasi",
        "tulasi": "Tulasi",
        "indian gooseberry": "Amla",
        "giloy": "Amruta Balli",
        "lemongrass": "Lemon Grass",
        "wood sorrel": "Wood Sorel",
        "neem": "Neem",
        "mango": "Mango",
        "mint": "Mint",
        "तुलसी": "Tulasi",
        "नीम": "Neem",
        "आम": "Mango",
        "पुदीना": "Mint",
        "आंवला": "Amla",
        "ಅಲೋವೆರಾ": "Aloevera",
        "ಪೇರಲ": "Guava",
        "ಸೀಬೆ": "Guava",
        "ಪಪ್ಪಾಯಿ": "Papaya",
        "ತುಳಸಿ": "Tulasi",
        "ಬೇವು": "Neem",
        "ಬೇವಿನ": "Neem",
        "ಮಾವು": "Mango",
        "ಮಾವಿನ": "Mango",
        "ಪುದೀನಾ": "Mint",
        "ನೆಲ್ಲಿಕಾಯಿ": "Amla"
    }

    found = None

    # Check aliases first, longest names first.
    for alias in sorted(aliases, key=len, reverse=True):
        if alias in search_text:
            found = aliases[alias]
            break

    # Check the plant database if no alias matched.
    if found is None:
        for plant in sorted(medicinal_uses, key=len, reverse=True):
            if plant.lower() in search_text:
                found = plant
                break

    # Handle greetings and general help.
    if any(word in question for word in (
        "hello", "hi", "hey", "namaste", "ನಮಸ್ಕಾರ", "ನಮಸ್ತೆ",
        "नमस्ते", "नमस्कार"
    )):
        response_en = (
            "Hello! I am ArogyaMitra. Ask me about a medicinal plant, "
            "its traditional uses, or plant disease information."
        )

    elif any(word in question for word in (
        "help", "what can you do", "who are you"
    )):
        response_en = (
            "I am ArogyaMitra, a medicinal-plant information chatbot. "
            "Ask about a plant name, its uses, or plant disease care."
        )

    elif found:
        uses = medicinal_uses.get(
            found, "No use information is available."
        )

        asks_disease = any(word in search_text for word in (
            "disease", "diseases", "infection", "spots", "symptom",
            "symptoms", "condition", "pest", "fungus", "fungal",
            "yellow leaves", "leaf problem", "care", "solution",
            "treatment", "ರೋಗ", "ರೋಗಗಳು", "ಕೀಟ", "ಹಳದಿ ಎಲೆ",
            "बीमारी", "रोग", "कीट", "पीली पत्तियां", "इलाज",
            "उपचार"
        ))

        if asks_disease and found in plant_diseases:
            info = plant_diseases[found]
            response_en = (
                f'{found} — {info["status"]}. '
                f'Details: {info["details"]}. '
                f'Suggested care: {info["solution"]}.'
            )

        elif asks_disease:
            response_en = (
                f"{found}: {uses} "
                "Specific disease information is not available in the "
                "chatbot database for this plant. You can use the image "
                "disease-detection feature for supported plants."
            )

        else:
            response_en = (
                f"{found}: {uses} "
                "These are general traditional-use descriptions, not "
                "medical advice. Please consult a qualified healthcare "
                "professional before using a plant medicinally."
            )

    else:
        response_en = (
            "I could not identify a plant in your question. Please include "
            'the plant name, for example: "What are the uses of Neem?" '
            "You can ask in English, Kannada, or Hindi."
        )

    # Translate the answer back into the user's language.
    response_translated = response_en

    if src_lang in ("kn", "hi"):
        try:
            translated_response = translate_text(
                response_en, "en", src_lang
            )
            if (
                isinstance(translated_response, str)
                and translated_response.strip()
            ):
                response_translated = translated_response
        except Exception:
            app.logger.exception("Response translation failed")

    return jsonify({
        "response": response_translated,
        "language": src_lang,
        "plant": found
    })


# ============================================================
# GROQ WHISPER SPEECH-TO-TEXT
# ============================================================

@app.route("/transcribe", methods=["POST"])
def transcribe():
    if groq_client is None:
        return jsonify({
            "error": "Groq API key is not configured. Check your .env file."
        }), 503

    if "audio" not in request.files:
        return jsonify({"error": "No audio file received"}), 400

    audio = request.files["audio"]

    if not audio.filename:
        return jsonify({"error": "Empty audio file"}), 400

    try:
        audio_bytes = audio.read()
        if not audio_bytes:
            return jsonify({"error": "No audio data received"}), 400

        language = {
            "en-IN": "en",
            "hi-IN": "hi",
            "kn-IN": "kn"
        }.get(request.form.get("language", "en-IN"), "en")

        transcription = groq_client.audio.transcriptions.create(
            file=(audio.filename, audio_bytes),
            model="whisper-large-v3",
            language=language
        )

        return jsonify({"text": transcription.text})

    except Exception:
        app.logger.exception("Audio transcription failed")
        return jsonify({
            "error": (
                "Transcription failed. Check your network, Groq API key, "
                "supported audio format, and API usage limits."
            )
        }), 500


if __name__ == "__main__":
    os.makedirs(os.path.join(BASE_DIR, "uploads"), exist_ok=True)
    app.run(host="127.0.0.1", port=5001, debug=False)
