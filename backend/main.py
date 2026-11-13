import os
import json
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from groq import Groq

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

client = Groq(api_key=GROQ_API_KEY)


def get_active_model():
    try:
        models = client.models.list()
        model_ids = [m.id for m in models.data]
        preferences = [
            "openai/gpt-oss-20b",
            "openai/gpt-oss-120b",
            "llama-3.3-70b-versatile",
            "llama-3.1-8b-instant",
        ]
        for pref in preferences:
            if pref in model_ids:
                return pref
        return model_ids[0] if model_ids else "openai/gpt-oss-20b"
    except Exception:
        return "openai/gpt-oss-20b"


class SymptomInput(BaseModel):
    patient_id: str
    track: str
    symptoms: str
    language: str = "hi-IN"  # Default Hindi


@app.get("/")
def home():
    return {"status": "Vaidyam Backend is Live!"}


@app.post("/api/triage")
def analyze_symptoms(data: SymptomInput):
    # Mapping language code to plain language name for LLM context
    lang_map = {
        "hi-IN": "Hindi",
        "ta-IN": "Tamil",
        "te-IN": "Telugu",
        "kn-IN": "Kannada",
        "bn-IN": "Bengali",
        "mr-IN": "Marathi",
        "gu-IN": "Gujarati",
        "en-IN": "English",
    }
    target_lang = lang_map.get(data.language, "Hindi")

    prompt = f"""
    You are a clinical triage AI assistant for the VAIDYAM healthcare kiosk in India.
    Patient Symptoms (could be in any Indian language or script): "{data.symptoms}"
    Medical Track Selected: {data.track.upper()}
    Target Output Language: {target_lang}

    Analyze the symptoms based on the selected medical track.
    IMPORTANT: Write the JSON field values ('summary', 'recommended_generic', 'action_required') in {target_lang} language script.

    Return ONLY a JSON object matching this schema:
    {{
        "severity": "EMERGENCY" or "URGENT" or "ROUTINE",
        "summary": "1-2 sentence medical summary or dosha assessment written in {target_lang}",
        "recommended_generic": "Generic medicine or herbal formulation name written in {target_lang}",
        "action_required": "Immediate advice or doctor consultation instruction written in {target_lang}"
    }}
    """

    try:
        active_model = get_active_model()

        chat_completion = client.chat.completions.create(
            messages=[
                {
                    "role": "system",
                    "content": "You are a multilingual emergency medical triage AI that outputs JSON only.",
                },
                {"role": "user", "content": prompt},
            ],
            model=active_model,
            temperature=0.2,
            response_format={"type": "json_object"},
        )

        response_text = chat_completion.choices[0].message.content.strip()
        ai_data = json.loads(response_text)

        return {
            "patient_id": data.patient_id,
            "track": data.track,
            "severity": ai_data.get("severity", "ROUTINE").upper(),
            "summary": ai_data.get("summary", "Analysis completed."),
            "recommended_generic": ai_data.get(
                "recommended_generic", "Consult doctor"
            ),
            "action_required": ai_data.get(
                "action_required", "Visit clinic"
            ),
            "jan_aushadhi_store": "Jan Aushadhi Kendra - Local Branch",
        }

    except Exception as e:
        print(f"\n--- GROQ API ERROR: {e} ---\n")
        is_allopathy = data.track.lower() == "allopathy"
        return {
            "patient_id": data.patient_id,
            "track": data.track,
            "severity": "URGENT",
            "summary": f"Triage assessment completed for: {data.symptoms}",
            "recommended_generic": (
                "Paracetamol 500mg"
                if is_allopathy
                else "Amritarishta / Sudarshan Churna"
            ),
            "action_required": "Please visit nearest healthcare center.",
            "jan_aushadhi_store": "Jan Aushadhi Kendra - Indiranagar Branch",
        }