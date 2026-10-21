# AI-Based Plant Identification and Disease Detection

An AI-powered web application designed to support medicinal plant identification, plant disease detection, and plant-related information through a multilingual chatbot.

## Features

- Identify medicinal plants from uploaded images.
- Predict plant diseases using trained deep-learning models.
- Ask plant-related questions through an AI chatbot.
- Interact in English, Kannada, and Hindi.
- Use voice input for speech-based interaction.
- Access the application through a Flask web interface.

## Technology Stack

- **Backend:** Python, Flask
- **Frontend:** HTML, CSS, JavaScript
- **Machine Learning:** TensorFlow / Keras, NumPy
- **Chatbot and speech recognition APIs:** Groq
- **Translation:** deep-translator

## Project Structure

```text
IEEE_internship/
├── app.py
├── Medicine_predict.py
├── requirements.txt
├── .env.example
├── README.md
├── ml_v2/
│   └── results/
│       ├── disease_model/
│       │   ├── best_disease_model.keras
│       │   └── class_names.json
│       └── plant_mobilenetv2/
│           ├── best_plant_mobilenetv2.keras
│           └── class_indices.json
├── static/
│   ├── css/
│   └── images/
└── templates/
    ├── home.html
    └── index.html
```

## Prerequisites

- Python 3.12 recommended, subject to TensorFlow compatibility with your operating system and architecture.
- pip
- A Groq API key for chatbot and voice-transcription features.
- Internet access for API calls and online translation.

## Installation and Setup

1. Clone this repository and enter the project directory:

   ```bash
   git clone <YOUR_GITHUB_REPOSITORY_URL>
   cd IEEE_internship
   ```

2. Create and activate a virtual environment:

   ```bash
   python3 -m venv .venv
   source .venv/bin/activate
   ```

3. Install the dependencies:

   ```bash
   python -m pip install --upgrade pip
   pip install -r requirements.txt
   ```

4. Create your local environment file:

   ```bash
   cp .env.example .env
   ```

5. Open `.env` and replace the placeholder with your own Groq API key:

   ```text
   GROQ_API_KEY=your_groq_api_key_here
   ```

   Keep `.env` private. Never commit your real API key to GitHub.

## Run the Application

Start the Flask application:

```bash
python app.py
```

Open the local address shown by Flask in your terminal. The application may use port 5001, depending on its configuration.

## Model Files

The trained model files and their class-label metadata are included in this repository. The application requires these files at the paths shown in the project structure above.

## API and Feature Availability

- Groq-powered chatbot and speech transcription require a valid API key and an internet connection.
- Translation may require an internet connection.
- Image prediction requires the included model files and compatible Python dependencies.

## Security

- Do not upload `.env` or any real API keys.
- Use `.env.example` as a template for environment configuration.
- Do not upload private datasets or credentials.

## License

No license has been specified yet. Add an appropriate license after confirming that you have permission to redistribute the source code, model weights, and other included assets.
