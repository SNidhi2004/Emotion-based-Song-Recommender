# Emotion-based-Song-Recommender

A song recommendation system that suggests songs as per the users feeling , using NLP and Deep Learning

Features

- Detects emotion : Given what user is feeling ,using a fine-tuned DistilBERT model
- Recommends songs : That match the detected emotion (as per the lyrics)
- Visualizes: Timeline chart for mood tracking and pie for overall mood visulaization
- Date-based mood history :  Filtering and Deletion
- Storage of Previous Moods : Locally in a csv file

---

##  How to Run

1. **Clone the repo**  
   ```bash
   git clone https://github.com/SNidhi2004/Emotion-based-Song-Recommender.git
   cd Emotion-based-Song-Recommender

2. **Create & activate a virtual environment**
   ```bash
   python -m venv venv
   venv\Scripts\activate

3. **Install Dependencies**
   ```bash
   pip install -r requirements.txt

4 **Run the app**
  ```bash
  streamlit run app/app.py
  ```

5 **For models Run**
  ```bash
  python src/train_model.py
```

**Project File Structure**
**.
├── app.py                  # Main Streamlit app
├── data/
│   └── mood_history.csv    # Mood history log
├── model/
│   ├── distilbert/         # HuggingFace model files
│   └── label_encoder.pkl   # Trained label encoder
├── src/
│   └── recommend_songs.py  # Song recommendation logic
├── requirements.txt
└── README.md**
