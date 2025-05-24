# src/predict_and_recommend.py

from transformers import DistilBertForSequenceClassification, DistilBertTokenizerFast
import torch
import joblib
from recommend_songs import load_song_dataset, recommend_songs

# Load model, tokenizer, and label encoder
tokenizer = DistilBertTokenizerFast.from_pretrained("model/distilbert")
model = DistilBertForSequenceClassification.from_pretrained("model/distilbert")
label_encoder = joblib.load("model/label_encoder.pkl")
songs_df = load_song_dataset()

def get_emotion(text):
    model.eval()
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
    with torch.no_grad():
        outputs = model(**inputs)
    logits = outputs.logits
    prediction = torch.argmax(logits, dim=1).item()
    return label_encoder.inverse_transform([prediction])[0]

def get_recommendations(text):
    emotion = get_emotion(text)
    print(f"🎯 Predicted Emotion: {emotion}\n")
    recs = recommend_songs(emotion, songs_df)
    if recs.empty:
        print("⚠️ No matching songs found.")
    else:
        print("🎵 Recommended Songs:")
        for _, row in recs.iterrows():
            print(f"- {row['SongName']} ({row['Feeling']})")

# Example CLI use
if __name__ == "__main__":
    user_input = input("Describe your feelings: ")
    get_recommendations(user_input)
