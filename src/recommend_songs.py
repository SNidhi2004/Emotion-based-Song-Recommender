# recommend_songs.py for getting and mapping the songs with emotion

import pandas as pd

def load_song_dataset(path='data/musicemotions.csv'):
    df = pd.read_csv(path)
    df['Feeling'] = df['Feeling'].str.lower().str.strip()
    return df

def recommend_songs(predicted_emotion, song_df, n=5):
    emotion = predicted_emotion.lower().strip()
    # Return all songs where 'Feeling' exactly matches the predicted emotion
    filtered = song_df[song_df['Feeling'] == emotion]
    return filtered.sample(n=min(n, len(filtered))) if not filtered.empty else pd.DataFrame()
