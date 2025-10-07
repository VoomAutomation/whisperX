import io
import gc
from contextlib import asynccontextmanager

import torch
import numpy as np
import soundfile as sf
import whisperx
from fastapi import FastAPI, File, UploadFile, Form
from fastapi.responses import JSONResponse


torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True


# Egyptian Arabic (ar-eg) phoneme mapping
EGYPTIAN_ARABIC_PHONEMES = {
    'ا': 'aː', 'أ': 'ʔa', 'إ': 'ʔi', 'آ': 'ʔaː', 'ؤ': 'ʔu', 'ئ': 'ʔi',
    'ء': 'ʔ', 'ب': 'b', 'ت': 't', 'ث': 's',  # ث -> s in Egyptian
    'ج': 'g',  # ج -> g in Egyptian (not dʒ)
    'ح': 'ħ', 'خ': 'x', 'د': 'd', 'ذ': 'z',  # ذ -> z in Egyptian
    'ر': 'r', 'ز': 'z', 'س': 's', 'ش': 'ʃ',
    'ص': 'sˤ', 'ض': 'dˤ', 'ط': 'tˤ', 'ظ': 'zˤ',  # ظ -> zˤ in Egyptian
    'ع': 'ʕ', 'غ': 'ɣ', 'ف': 'f', 
    'ق': 'ʔ',  # ق -> ʔ in Egyptian (glottal stop)
    'ك': 'k', 'ل': 'l', 'م': 'm', 'ن': 'n',
    'ه': 'h', 'و': 'w', 'ي': 'j', 'ى': 'aː', 'ة': 'a',
    'َ': 'a', 'ِ': 'e',  # ِ -> e in Egyptian
    'ُ': 'o',  # ُ -> o in Egyptian
    'ً': 'an', 'ٍ': 'en', 'ٌ': 'on',
}

# Saudi Arabic (ar-sa) phoneme mapping
SAUDI_ARABIC_PHONEMES = {
    'ا': 'aː', 'أ': 'ʔa', 'إ': 'ʔi', 'آ': 'ʔaː', 'ؤ': 'ʔu', 'ئ': 'ʔi',
    'ء': 'ʔ', 'ب': 'b', 'ت': 't', 'ث': 'θ', 'ج': 'dʒ', 'ح': 'ħ', 
    'خ': 'x', 'د': 'd', 'ذ': 'ð', 'ر': 'r', 'ز': 'z', 'س': 's', 
    'ش': 'ʃ', 'ص': 'sˤ', 'ض': 'dˤ', 'ط': 'tˤ', 'ظ': 'ðˤ', 'ع': 'ʕ', 
    'غ': 'ɣ', 'ف': 'f', 'ق': 'q', 'ك': 'k', 'ل': 'l', 'م': 'm', 
    'ن': 'n', 'ه': 'h', 'و': 'w', 'ي': 'j', 'ى': 'aː', 'ة': 'h',
    'َ': 'a', 'ِ': 'i', 'ُ': 'u', 'ً': 'an', 'ٍ': 'in', 'ٌ': 'un',
}

# Standard Arabic (ar) phoneme mapping
STANDARD_ARABIC_PHONEMES = {
    'ا': 'aː', 'أ': 'ʔa', 'إ': 'ʔi', 'آ': 'ʔaː', 'ؤ': 'ʔu', 'ئ': 'ʔi',
    'ء': 'ʔ', 'ب': 'b', 'ت': 't', 'ث': 'θ', 'ج': 'dʒ', 'ح': 'ħ', 
    'خ': 'x', 'د': 'd', 'ذ': 'ð', 'ر': 'r', 'ز': 'z', 'س': 's', 
    'ش': 'ʃ', 'ص': 'sˤ', 'ض': 'dˤ', 'ط': 'tˤ', 'ظ': 'ðˤ', 'ع': 'ʕ', 
    'غ': 'ɣ', 'ف': 'f', 'ق': 'q', 'ك': 'k', 'ل': 'l', 'م': 'm', 
    'ن': 'n', 'ه': 'h', 'و': 'w', 'ي': 'j', 'ى': 'aː', 'ة': 'h',
    'َ': 'a', 'ِ': 'i', 'ُ': 'u', 'ً': 'an', 'ٍ': 'in', 'ٌ': 'un',
}

# English - We'll use a phonetic dictionary for better accuracy
# This is a simplified mapping for common phonemes
ENGLISH_PHONEME_GROUPS = {
    'vowels': ['a', 'e', 'i', 'o', 'u', 'A', 'E', 'I', 'O', 'U'],
    'consonants': ['b', 'c', 'd', 'f', 'g', 'h', 'j', 'k', 'l', 'm', 
                   'n', 'p', 'q', 'r', 's', 't', 'v', 'w', 'x', 'y', 'z',
                   'B', 'C', 'D', 'F', 'G', 'H', 'J', 'K', 'L', 'M',
                   'N', 'P', 'Q', 'R', 'S', 'T', 'V', 'W', 'X', 'Y', 'Z']
}

# Dialect to phoneme map and language code
DIALECT_CONFIG = {
    'ar': {'phonemes': STANDARD_ARABIC_PHONEMES, 'lang': 'ar', 'use_char_map': True},
    'ar-eg': {'phonemes': EGYPTIAN_ARABIC_PHONEMES, 'lang': 'ar', 'use_char_map': True},
    'ar-sa': {'phonemes': SAUDI_ARABIC_PHONEMES, 'lang': 'ar', 'use_char_map': True},
    'en': {'phonemes': ENGLISH_PHONEME_GROUPS, 'lang': 'en', 'use_char_map': False},
}


@asynccontextmanager
async def lifespan(app: FastAPI):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    compute_type = "float16" if device == "cuda" else "int8"

    print(f"Loading model on {device} with compute_type={compute_type}")
    
    app.state.whisper_model = whisperx.load_model(
        "tiny", device, compute_type=compute_type
    )
    
    # Pre-load alignment models for both Arabic and English
    app.state.align_models = {}
    print("Loading alignment models...")
    app.state.align_models['ar'], app.state.align_metadata_ar = whisperx.load_align_model(
        language_code="ar", device=device
    )
    app.state.align_models['en'], app.state.align_metadata_en = whisperx.load_align_model(
        language_code="en", device=device
    )
    
    app.state.device = device
    print("Models loaded successfully!")
    
    yield
    
    del app.state.whisper_model
    del app.state.align_models
    gc.collect()


app = FastAPI(lifespan=lifespan)


@app.post("/transcribe")
async def transcribe_audio(
    file: UploadFile = File(...),
    dialect: str = Form(default="en"),
    language: str = Form(default="en")
):
    try:
        # Validate dialect
        if dialect not in DIALECT_CONFIG:
            return JSONResponse(
                content={"error": f"Unsupported dialect: {dialect}. Supported: ar, ar-eg, ar-sa, en"},
                status_code=400
            )
        
        config = DIALECT_CONFIG[dialect]
        phoneme_map = config['phonemes']
        use_char_map = config['use_char_map']
        
        # Determine language to use - prioritize explicit language parameter
        if language and language != "":
            lang_code = language
            print(f"Using explicit language: {lang_code}")
        else:
            lang_code = config['lang']
            print(f"Using dialect language: {lang_code}")
        
        # Get appropriate alignment model and metadata BEFORE transcription
        align_model = app.state.align_models.get(lang_code)
        if lang_code == 'ar':
            align_metadata = app.state.align_metadata_ar
        else:
            align_metadata = app.state.align_metadata_en
        
        if not align_model:
            return JSONResponse(
                content={"error": f"No alignment model for language: {lang_code}"},
                status_code=400
            )
        
        # Load audio
        audio_bytes = await file.read()
        audio_buffer = io.BytesIO(audio_bytes)
        waveform, sample_rate = sf.read(audio_buffer)

        if waveform.ndim > 1:
            waveform = np.mean(waveform, axis=1)

        if sample_rate != 16000:
            import librosa
            waveform = librosa.resample(
                waveform, orig_sr=sample_rate, target_sr=16000
            )

        audio = waveform.astype(np.float32)

        # Transcribe with FORCED language (don't let it auto-detect)
        print(f"Transcribing with language={lang_code}")
        result = app.state.whisper_model.transcribe(
            audio, 
            batch_size=64,  # Reduced batch size for stability
            language=lang_code  # ALWAYS force the language
        )
        
        detected_lang = result.get("language", lang_code)
        print(f"Detected language: {detected_lang}, Using language: {lang_code}")
        
        # Don't change alignment model based on detection - stick with user's choice
        # This prevents the "treating English as Arabic" bug

        # Align with the CORRECT language model
        print(f"Aligning with {lang_code} model...")
        result_aligned = whisperx.align(
            result["segments"],
            align_model,
            align_metadata,
            audio,
            app.state.device,
            return_char_alignments=True
        )

        phoneme_timeline = []

        for segment in result_aligned["segments"]:
            segment_text = segment.get("text", "")
            print(f"Segment: {segment_text}")
            
            chars = segment.get("chars", [])
            
            for i, char_data in enumerate(chars):
                char = char_data["char"]
                
                # Skip whitespace and punctuation
                if char == ' ' or char in '.,!?;:\'"':
                    continue
                
                start = char_data.get("start")
                end = char_data.get("end")
                
                # Skip if no timing info
                if start is None or end is None:
                    continue

                # Determine which phoneme mapping to use
                if lang_code == 'ar' and use_char_map:
                    # Arabic phoneme mapping
                    if char == 'و':
                        phoneme = 'uː' if i > 0 and chars[i-1]["char"] in ['ُ', 'ِ', 'َ'] else 'w'
                    elif char == 'ي':
                        phoneme = 'iː' if i > 0 and chars[i-1]["char"] in ['ُ', 'ِ', 'َ'] else 'j'
                    elif char in phoneme_map:
                        phoneme = phoneme_map[char]
                        if not phoneme:
                            continue
                    else:
                        phoneme = char
                else:
                    # For English, use the character itself as phoneme
                    phoneme = char.lower()

                phoneme_timeline.append({
                    "phoneme": phoneme,
                    "start": round(start, 3),
                    "end": round(end, 3),
                    "duration": round(end - start, 3)
                })

        print(f"Generated {len(phoneme_timeline)} phonemes")
        
        gc.collect()
        
        return JSONResponse(content={
            "phonemes": phoneme_timeline,
            "detected_language": detected_lang,
            "used_language": lang_code,
            "text": " ".join([s.get("text", "") for s in result_aligned["segments"]])
        }, media_type="application/json")

    except Exception as e:
        import traceback
        error_details = traceback.format_exc()
        print(f"Error: {error_details}")
        return JSONResponse(content={
            "error": str(e),
            "details": error_details
        }, status_code=500)