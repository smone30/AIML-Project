import joblib
import neattext.functions as nfx


model = joblib.load("emotion_model.pkl")
vectorizer = joblib.load("tfidf_vectorizer.pkl")

def predict_emotion(text):
    cleaned = nfx.remove_special_characters(text).lower()  
    vect = vectorizer.transform([cleaned])
    prediction = model.predict(vect)
    return prediction[0]

print("TEXT-BASED EMOTION DETECTOR (Type 'exit' to stop)")
while True:
    user_input = input("You: ")
    if user_input.lower() == 'exit':
        break
    emotion = predict_emotion(user_input)
    print("Predicted Emotion:", emotion)
