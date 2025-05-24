import streamlit as st
import joblib
import torch
import os
import pandas as pd
import urllib.parse
import plotly.express as px
from transformers import DistilBertForSequenceClassification, DistilBertTokenizerFast
from datetime import datetime
import sys

# Add src to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.recommend_songs import load_song_dataset, recommend_songs

# File to store mood history
HISTORY_FILE = "data/mood_history.csv"

# Emotion mapping
EMOTION_MAP = {
    'sadness': ['sad', 'dreamy'],
    'anger': ['annoying', 'anxious'],
    'love': ['happy', 'joyful', 'dreamy'],
    'surprise': ['amusing', 'happy'],
    'fear': ['anxious'],
    'joy': ['happy', 'joyful', 'energizing', 'amusing']
}

# Directly link to Spotify
def create_spotify_search_link(song_name):
    return "https://open.spotify.com/search/" + urllib.parse.quote(song_name)

# Save the emotion history in a csv file
def save_mood_history(input_text, predicted_emotion):
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    new_entry = pd.DataFrame([[now, input_text, predicted_emotion]], columns=['timestamp', 'text', 'emotion'])

    if os.path.exists(HISTORY_FILE):
        old = pd.read_csv(HISTORY_FILE)
        new_entry = pd.concat([old, new_entry])
    new_entry.to_csv(HISTORY_FILE, index=False)

# Load model and data
tokenizer = DistilBertTokenizerFast.from_pretrained("model/distilbert")
model = DistilBertForSequenceClassification.from_pretrained("model/distilbert")
label_encoder = joblib.load("model/label_encoder.pkl")
songs_df = load_song_dataset()

# Streamlit app
st.title("Emotion-Based Song Recommender")
user_input = st.text_area("How are you feeling?")

if st.button("Get Recommendations"):
    if user_input.strip():
        model.eval()
        inputs = tokenizer(user_input, return_tensors="pt", truncation=True, padding=True)
        with torch.no_grad():
            outputs = model(**inputs)
        logits = outputs.logits
        prediction = torch.argmax(logits, dim=1).item()
        predicted_emotion = label_encoder.inverse_transform([prediction])[0]

        save_mood_history(user_input, predicted_emotion)

        mapped_emotions = EMOTION_MAP.get(predicted_emotion, [])
        st.markdown(f"**Predicted Emotion:** `{predicted_emotion}`")

        if mapped_emotions:
            st.markdown(f"**Mapped to:** {', '.join(mapped_emotions)}")

            all_recs = pd.DataFrame()
            for emotion in mapped_emotions:
                recs = recommend_songs(emotion, songs_df)
                if not recs.empty:
                    all_recs = pd.concat([all_recs, recs])

            if not all_recs.empty:
                unique_recs = all_recs.drop_duplicates(subset=['SongName'])
                st.write("Recommended Songs:")
                for _, row in unique_recs.head(10).iterrows():
                    song_name = row['SongName']
                    link = create_spotify_search_link(song_name)
                    st.markdown(f"- [{song_name} ({row['Feeling']})]({link})")

                csv = unique_recs.to_csv(index=False)
                st.download_button("Download Playlist as .csv File", csv, "playlist.csv", "text/csv")
            else:
                st.warning("No matching songs found for these emotions.")
        else:
            st.warning(f"No emotion mapping found for '{predicted_emotion}'")
    else:
        st.warning("Please type a sentence.")

# Mood history & visualizations
if os.path.exists(HISTORY_FILE):
      st.write("**Mood History**")
      history = pd.read_csv(HISTORY_FILE)
      if not history.empty:
        
         st.subheader("Mood Entries")

# Pagination
         page_size = 10
         total_rows = len(history)
         total_pages = (total_rows - 1) // page_size + 1
         page = st.number_input("Page", 1, total_pages, step=1)

         start_idx = (page - 1) * page_size
         end_idx = start_idx + page_size

# Show paginated table
         st.dataframe(history.iloc[start_idx:end_idx])

         history['timestamp'] = pd.to_datetime(history['timestamp'])
      
         st.subheader("**Mood Timeline**")
         fig_line = px.line(history, x='timestamp', y='emotion', markers=True, title="Mood over Time")
         st.plotly_chart(fig_line, use_container_width=True)

         st.subheader("**Emotion Distribution over pie chart**")
         fig_pie = px.pie(history, names='emotion')
         st.plotly_chart(fig_pie, use_container_width=True)
      else:
         st.info("History file exists but is empty.")
else:
    st.info("No mood history file found.")

st.subheader("**Select and Delete Multiple Mood History Entries**")

with st.expander("Delete Entries from Mood History", expanded=False):
    history['date'] = history['timestamp'].dt.date

    # Date selection
    selected_date = st.date_input("Select date to filter entries")

    # All button for showing all entries at once
    all_button = st.button("Show All Entries")

    # Determine which entries to show as per the field choosen above
    if all_button:
        filtered_history = history
        st.success("Showing all entries")
    else:
        filtered_history = history[history['date'] == selected_date]
        if filtered_history.empty:
            st.info(f"No entries found for {selected_date}")

    if not filtered_history.empty:
        selected_indices = []
        for i, row in filtered_history.iterrows():
            preview = row["text"][:40] + "..." if len(row["text"]) > 40 else row["text"]
            label = f"[{i}] {row['timestamp']} — {row['emotion']} — {preview}"
            if st.checkbox(label, key=f"del_{i}"):
                selected_indices.append(i)

        if selected_indices:
            if st.button("Delete Selected Entries"):
                history = history.drop(selected_indices).reset_index(drop=True)
                history.to_csv(HISTORY_FILE, index=False)
                st.rerun()