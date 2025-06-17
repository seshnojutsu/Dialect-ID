import whisper
import torch.nn.functional as F
import torch

# Load Whisper ASR model
asr_model = whisper.load_model("small")  # or "base", "medium", "large"

# Transcribe Arabic audio
def transcribe_audio(file_path):
    result = asr_model.transcribe(file_path, language='ar')
    return result["text"]

# --- Audio Classification Section ---

import torchaudio
torchaudio.set_audio_backend("soundfile")  # <-- Add it right after this import

from transformers import AutoProcessor, AutoModelForAudioClassification, Wav2Vec2FeatureExtractor

model_id = "badrex/mms-300m-arabic-dialect-identifier"

# Use only the feature extractor (not processor)
feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(model_id)
model = AutoModelForAudioClassification.from_pretrained(model_id)


def normalize_waveform(waveform):
    # Normalize to [-1.0, 1.0]
    max_val = waveform.abs().max()
    if max_val > 0:
        waveform = waveform / max_val
    return waveform


def load_audio(file_path):
    waveform, sample_rate = torchaudio.load(file_path)

    # Resample if needed
    if sample_rate != 16000:
        waveform = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=16000)(waveform)
    
    # Convert to mono
    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)

    # Normalize
    waveform = normalize_waveform(waveform)
    
    return waveform

def predict_dialect(file_path):
    waveform = load_audio(file_path)
    
    inputs = feature_extractor(waveform.squeeze().numpy(), sampling_rate=16000, return_tensors="pt")
    
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        predicted_class = torch.argmax(logits, dim=1).item()
        predicted_label = model.config.id2label[predicted_class]
        confidence = torch.nn.functional.softmax(logits, dim=1)[0][predicted_class].item()
    
    return predicted_label, round(confidence, 3)

# Example usage
if __name__ == "__main__":
    audio_file = "/Users/seshnojutsu/Downloads/converted_6.wav"
    transcription = transcribe_audio(audio_file)
    dialect, confidence = predict_dialect(audio_file)
    print(f"Transcription: {transcription}")
    print(f"Predicted Dialect: {dialect} (Confidence: {confidence})")

