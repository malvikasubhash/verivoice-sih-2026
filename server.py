
import os
import uuid
import sqlite3
import subprocess
from datetime import datetime

import numpy as np
import soundfile as sf
import torch

from flask import Flask, request, jsonify
from flask_cors import CORS

from transformers import (
    AutoFeatureExtractor,
    AutoModelForAudioClassification
)

from silero_vad import (
    load_silero_vad,
    get_speech_timestamps
)

# ============================================================
# SETTINGS
# ============================================================

MODEL_ID = "HyperMoon/wav2vec2-base-960h-finetuned-deepfake"

MODEL_NAME = MODEL_ID

SAMPLE_RATE = 16000

HIGH_THRESHOLD = 0.99
SUSPICIOUS_THRESHOLD = 0.30

DB_PATH = "/content/verivoice.db"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

app = Flask(__name__)
CORS(app)


# ============================================================
# DATABASE
# ============================================================

def get_db():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_db():

    conn = get_db()

    conn.execute("""
        CREATE TABLE IF NOT EXISTS analyses (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            filename TEXT,
            risk TEXT,
            spoof_probability REAL,
            bonafide_probability REAL,
            confidence REAL,
            message TEXT,
            created_at TEXT
        )
    """)

    conn.execute("""
        CREATE TABLE IF NOT EXISTS reports (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            phone TEXT,
            reason TEXT,
            details TEXT,
            created_at TEXT
        )
    """)

    conn.execute("""
        CREATE TABLE IF NOT EXISTS feedback (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            rating INTEGER,
            category TEXT,
            comment TEXT,
            created_at TEXT
        )
    """)

    conn.commit()
    conn.close()


init_db()


# ============================================================
# LOAD ORIGINAL MODEL DIRECTLY
# ============================================================

print("🧠 Loading original HyperMoon model...")

processor = AutoFeatureExtractor.from_pretrained(
    MODEL_ID
)

model = AutoModelForAudioClassification.from_pretrained(
    MODEL_ID
)

model.to(DEVICE)
model.eval()

print("✅ AI model loaded on:", DEVICE)

print("Labels:", model.config.id2label)


# ============================================================
# SPEECH DETECTOR
# ============================================================

print("🎙 Loading speech detector...")

vad_model = load_silero_vad()

print("✅ Speech detector loaded.")


# ============================================================
# HEALTH
# ============================================================

@app.get("/api/health")
def health():

    return jsonify({
        "status": "ok",
        "service": "VeriVoice AI",
        "model": MODEL_NAME,
        "device": DEVICE,
        "high_threshold": HIGH_THRESHOLD,
        "suspicious_threshold": SUSPICIOUS_THRESHOLD
    })


# ============================================================
# CONVERT UPLOADED FILE TO WAV
# ============================================================

def convert_to_wav(uploaded):

    uid = uuid.uuid4().hex

    input_path = f"/tmp/verivoice_{uid}"

    output_path = f"/tmp/verivoice_{uid}.wav"

    uploaded.save(input_path)

    command = [
        "ffmpeg",
        "-y",
        "-i",
        input_path,
        "-ac",
        "1",
        "-ar",
        str(SAMPLE_RATE),
        "-sample_fmt",
        "s16",
        output_path
    ]

    result = subprocess.run(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE
    )

    try:
        os.remove(input_path)
    except:
        pass

    if result.returncode != 0:
        raise RuntimeError(
            "Unable to read the audio file."
        )

    return output_path


# ============================================================
# DIRECT MODEL PREDICTION
# ============================================================

def predict(audio):

    inputs = processor(
        audio,
        sampling_rate=SAMPLE_RATE,
        return_tensors="pt"
    )

    inputs = {
        key: value.to(DEVICE)
        for key, value in inputs.items()
    }

    with torch.no_grad():

        outputs = model(
            **inputs
        )

        probs = torch.softmax(
            outputs.logits,
            dim=-1
        )[0]

    labels = model.config.id2label

    spoof_index = None
    bonafide_index = None

    for index, label in labels.items():

        label_text = str(
            label
        ).lower()

        if "spoof" in label_text:
            spoof_index = int(index)

        if (
            "bonafide" in label_text
            or "genuine" in label_text
        ):
            bonafide_index = int(index)

    if spoof_index is None:
        spoof_index = 1

    if bonafide_index is None:
        bonafide_index = 0

    spoof = float(
        probs[spoof_index].cpu()
    )

    bonafide = float(
        probs[bonafide_index].cpu()
    )

    return spoof, bonafide


# ============================================================
# ANALYZE
# ============================================================

@app.post("/api/analyze")
def analyze():

    if "audio" not in request.files:

        return jsonify({
            "error": "No audio file received."
        }), 400

    uploaded = request.files["audio"]

    if not uploaded.filename:

        return jsonify({
            "error": "Audio filename is missing."
        }), 400

    wav_path = None

    try:

        # ----------------------------------------------------
        # CONVERT
        # ----------------------------------------------------

        wav_path = convert_to_wav(
            uploaded
        )

        audio, sr = sf.read(
            wav_path,
            dtype="float32"
        )

        if audio.ndim > 1:

            audio = np.mean(
                audio,
                axis=1
            )

        if sr != SAMPLE_RATE:

            raise RuntimeError(
                "Audio conversion failed."
            )

        if len(audio) == 0:

            raise RuntimeError(
                "No usable audio detected."
            )

        # ----------------------------------------------------
        # NORMALIZE
        # ----------------------------------------------------

        peak = float(
            np.max(
                np.abs(audio)
            )
        )

        if peak > 0:
            audio = audio / peak

        duration = (
            len(audio) / SAMPLE_RATE
        )

        if duration < 0.5:

            return jsonify({

                "success": True,
                "filename": uploaded.filename,

                "risk": "NON_SPEECH",
                "analysis_type": "non_speech",

                "spoof_probability": None,
                "bonafide_probability": None,
                "confidence": None,

                "message":
                    "The recording is too short "
                    "for reliable voice analysis."
            })

        # ----------------------------------------------------
        # SPEECH DETECTION
        # ----------------------------------------------------

        waveform = torch.tensor(
            audio,
            dtype=torch.float32
        )

        segments = get_speech_timestamps(
            waveform,
            vad_model,
            sampling_rate=SAMPLE_RATE,
            threshold=0.50,
            min_speech_duration_ms=250,
            min_silence_duration_ms=150
        )

        speech_samples = sum(
            int(x["end"]) -
            int(x["start"])
            for x in segments
        )

        speech_duration = (
            speech_samples /
            SAMPLE_RATE
        )

        speech_ratio = (
            speech_duration /
            duration
            if duration > 0
            else 0
        )

        # ----------------------------------------------------
        # MUSIC / NON-SPEECH
        # ----------------------------------------------------

        if (
            speech_duration < 0.75
            or speech_ratio < 0.15
        ):

            message = (
                "Not enough human speech was detected. "
                "VeriVoice does not assign a voice-cloning "
                "risk verdict to this recording."
            )

            created_at = datetime.now().isoformat(
                timespec="seconds"
            )

            conn = get_db()

            conn.execute("""
                INSERT INTO analyses (
                    filename,
                    risk,
                    spoof_probability,
                    bonafide_probability,
                    confidence,
                    message,
                    created_at
                )
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """, (
                uploaded.filename,
                "NON_SPEECH",
                None,
                None,
                None,
                message,
                created_at
            ))

            conn.commit()
            conn.close()

            return jsonify({

                "success": True,
                "filename": uploaded.filename,

                "risk": "NON_SPEECH",
                "analysis_type": "non_speech",

                "spoof_probability": None,
                "bonafide_probability": None,
                "confidence": None,

                "speech_duration": round(
                    speech_duration,
                    2
                ),

                "speech_ratio": round(
                    speech_ratio,
                    3
                ),

                "message": message,

                "timestamp": created_at
            })

        # ----------------------------------------------------
        # LIMIT LONG RECORDINGS
        # ----------------------------------------------------

        audio = audio[
            :30 * SAMPLE_RATE
        ]

        # ----------------------------------------------------
        # AI
        # ----------------------------------------------------

        spoof, bonafide = predict(
            audio
        )

        confidence = max(
            spoof,
            bonafide
        )

        # ----------------------------------------------------
        # RISK
        # ----------------------------------------------------

        if spoof >= HIGH_THRESHOLD:

            risk = "HIGH"

            message = (
                "Strong synthetic-voice characteristics "
                "were detected. Verify the caller independently "
                "before sharing sensitive information."
            )

        elif spoof >= SUSPICIOUS_THRESHOLD:

            risk = "SUSPICIOUS"

            message = (
                "The voice contains uncertain characteristics. "
                "Use independent verification before sensitive actions."
            )

        else:

            risk = "LOW"

            message = (
                "No strong synthetic-voice signal was detected "
                "in this sample."
            )

        # ----------------------------------------------------
        # ACTIVITY
        # ----------------------------------------------------

        created_at = datetime.now().isoformat(
            timespec="seconds"
        )

        conn = get_db()

        conn.execute("""
            INSERT INTO analyses (
                filename,
                risk,
                spoof_probability,
                bonafide_probability,
                confidence,
                message,
                created_at
            )
            VALUES (?, ?, ?, ?, ?, ?, ?)
        """, (
            uploaded.filename,
            risk,
            spoof,
            bonafide,
            confidence,
            message,
            created_at
        ))

        conn.commit()
        conn.close()

        # ----------------------------------------------------
        # RESPONSE
        # ----------------------------------------------------

        return jsonify({

            "success": True,

            "filename": uploaded.filename,

            "risk": risk,
            "analysis_type": "speech",

            "spoof_probability": spoof,
            "bonafide_probability": bonafide,

            "confidence": confidence,

            "spoof_percentage": round(
                spoof * 100,
                2
            ),

            "confidence_percentage": round(
                confidence * 100,
                2
            ),

            "speech_duration": round(
                speech_duration,
                2
            ),

            "speech_ratio": round(
                speech_ratio,
                3
            ),

            "message": message,

            "timestamp": created_at
        })

    except Exception as e:

        return jsonify({
            "error": "Voice analysis failed.",
            "details": str(e)
        }), 500

    finally:

        if wav_path:

            try:
                os.remove(wav_path)
            except:
                pass


# ============================================================
# NUMBER CHECK
# ============================================================

@app.get("/api/number/<path:phone>")
def number_check(phone):

    normalized = "".join(
        c for c in phone
        if c.isdigit() or c == "+"
    )

    conn = get_db()

    rows = conn.execute("""
        SELECT reason, details, created_at
        FROM reports
        WHERE phone = ?
        ORDER BY id DESC
    """, (normalized,)).fetchall()

    conn.close()

    return jsonify({
        "phone": normalized,
        "report_count": len(rows),
        "reports": [dict(x) for x in rows]
    })


# ============================================================
# REPORT
# ============================================================

@app.post("/api/report")
def report():

    data = request.get_json(
        silent=True
    ) or {}

    phone = str(
        data.get("phone", "")
    ).strip()

    reason = str(
        data.get("reason", "")
    ).strip()

    details = str(
        data.get("details", "")
    ).strip()

    if not phone:

        return jsonify({
            "error": "Phone number is required."
        }), 400

    conn = get_db()

    conn.execute("""
        INSERT INTO reports (
            phone,
            reason,
            details,
            created_at
        )
        VALUES (?, ?, ?, ?)
    """, (
        phone,
        reason,
        details,
        datetime.now().isoformat(
            timespec="seconds"
        )
    ))

    conn.commit()
    conn.close()

    return jsonify({
        "success": True
    })


# ============================================================
# ACTIVITY
# ============================================================

@app.get("/api/activity")
def activity():

    conn = get_db()

    rows = conn.execute("""
        SELECT
            filename,
            risk,
            spoof_probability,
            bonafide_probability,
            confidence,
            message,
            created_at
        FROM analyses
        ORDER BY id DESC
        LIMIT 100
    """).fetchall()

    conn.close()

    return jsonify({
        "analyses": [dict(x) for x in rows]
    })


# ============================================================
# FEEDBACK
# ============================================================

@app.post("/api/feedback")
def feedback():

    data = request.get_json(
        silent=True
    ) or {}

    conn = get_db()

    conn.execute("""
        INSERT INTO feedback (
            rating,
            category,
            comment,
            created_at
        )
        VALUES (?, ?, ?, ?)
    """, (
        int(data.get("rating", 0)),
        str(data.get("category", "")),
        str(data.get("comment", "")),
        datetime.now().isoformat(
            timespec="seconds"
        )
    ))

    conn.commit()
    conn.close()

    return jsonify({
        "success": True
    })


# ============================================================
# START
# ============================================================

if __name__ == "__main__":

    print("============================================")
    print("VERIVOICE STABLE BACKEND")
    print("============================================")
    print("Model:", MODEL_NAME)
    print("Device:", DEVICE)
    print("HIGH threshold:", HIGH_THRESHOLD)
    print("SUSPICIOUS threshold:", SUSPICIOUS_THRESHOLD)

    app.run(
        host="0.0.0.0",
        port=8000,
        threaded=True
    )
