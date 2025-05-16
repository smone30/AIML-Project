import whisper
import joblib
import neattext.functions as nfx


asr_model = whisper.load_model("base")


emotion_model = joblib.load("emotion_model.pkl")
vectorizer = joblib.load("tfidf_vectorizer.pkl")

def transcribe_and_predict(audio_path):
    
    result = asr_model.transcribe(audio_path)
    text = result['text']
    print("Transcribed Text:", text)

 
    cleaned = nfx.remove_special_characters(text).lower()


    vect_text = vectorizer.transform([cleaned])
    emotion = emotion_model.predict(vect_text)[0]
    print("Detected Emotion:", emotion)

# Example 
if __name__ == "__main__":
    transcribe_and_predict("Audio/Audio 2.mp3")  # Replace with  audio file path
