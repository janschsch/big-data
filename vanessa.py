import time
import json
import pandas as pd
import numpy as np
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from transformers import pipeline
import matplotlib.pyplot as plt
import streamlit as st

# Funktion zum Abrufen von Kommentaren
def get_comments(query, max_videos=100, max_comments_per_video=200):
    API_KEY = 'AIzaSyAUIf2y-Sos1m5S5UMvtBeiGD7WbjyT0L8'

    youtube = build('youtube', 'v3', developerKey=API_KEY)
    videos = []
    next_page_token = None

    while len(videos) < max_videos:
        request = youtube.search().list(
            q=query,
            part='id,snippet',
            type='video',
            order='viewCount',
            maxResults=50,
            pageToken=next_page_token
        )
        response = request.execute()

        for item in response['items']:
            video_id = item['id']['videoId']
            video_title = item['snippet']['title']
            videos.append({'video_id': video_id, 'video_title': video_title})

        next_page_token = response.get('nextPageToken')
        if not next_page_token:
            break
        time.sleep(0.1)

    videos = videos[:max_videos]
    all_comments = []

    for idx, video in enumerate(videos):
        video_id = video['video_id']
        video_title = video['video_title']
        next_page_token = None
        comments = []

        while len(comments) < max_comments_per_video:
            try:
                request = youtube.commentThreads().list(
                    part='snippet',
                    videoId=video_id,
                    maxResults=100,
                    pageToken=next_page_token,
                    textFormat='plainText',
                    order='relevance'
                )
                response = request.execute()

                for item in response['items']:
                    comment = item['snippet']['topLevelComment']['snippet']
                    comment_text = comment['textDisplay']
                    published_at = comment['publishedAt']
                    like_count = comment['likeCount']
                    comments.append({
                        'video_id': video_id,
                        'published_at': published_at,
                        'like_count': like_count,
                        'comment': comment_text
                    })

                next_page_token = response.get('nextPageToken')
                if not next_page_token:
                    break
                time.sleep(0.1)

            except HttpError as e:
                error_content = json.loads(e.content.decode('utf-8'))
                error_reason = error_content['error']['errors'][0]['reason']
                if error_reason == 'commentsDisabled':
                    st.write(f"Comments are disabled for video {video_id}.")
                else:
                    st.write(f"An HTTP error occurred while fetching comments for video {video_id}: {e}")
                break

        all_comments.extend(comments[:max_comments_per_video])
        time.sleep(0.1)

    df = pd.DataFrame(all_comments)
    df = df[['published_at', 'like_count', 'comment']]
    df['published_at'] = pd.to_datetime(df['published_at'])
    return df

# Funktion zur Verarbeitung und Darstellung der Daten
def process_and_visualize_comments(df, n_clusters=5):
    # 1. Clustering
    categories = {
    "Design": ["design", "style", "aesthetics", "look", "appearance"],
    "Camera": ["camera", "photo", "image", "picture", "zoom"],
    "Battery": ["battery", "charge", "charging time", "power", "energy"],
    "Performance": ["performance", "speed", "processor", "ram", "cpu"],
    "Price": ["price", "cost", "expensive", "cheap", "affordable", "value"]
    }

    def calculate_probabilities(comment):
        probabilities = {}
        comment_lower = comment.lower()
        for category, keywords in categories.items():
            probabilities[category] = sum(comment_lower.count(keyword) for keyword in keywords)
        return probabilities

    def assign_best_category(comment):
        probabilities = calculate_probabilities(comment)
        best_category = max(probabilities, key=probabilities.get)
        return best_category

    # Kommentaren Kategorien zuweisen
    df["category"] = df["comment"].apply(assign_best_category)

    # 2. Sentiment-Analyse: Kombiniert BERT und Emoji-Sentiment
    sentiment_pipeline = pipeline("sentiment-analysis", model="nlptown/bert-base-multilingual-uncased-sentiment")
    emoji_sentiment_data = pd.read_csv("Emoji_Sentiment_Data_v1.0.csv")

    def combined_sentiment(comment, emojis):
       
        # Text-Sentiment mit BERT berechnen
        text_sentiment = sentiment_pipeline(comment)[0]['score']

        # Emojis analysieren
       
        if not emojis:
            return text_sentiment  # Kein Emoji, nur Textsentiment

        emoji_scores = []
        for emoji_char in emojis:
            sentiment_row = emoji_sentiment_data[emoji_sentiment_data["Emoji"] == emoji_char]
            if not sentiment_row.empty:
                positive_score = sentiment_row["Positive (%)"].values[0]
                negative_score = sentiment_row["Negative (%)"].values[0]
                neutral_score = sentiment_row["Neutral (%)"].values[0]

                if positive_score > negative_score and positive_score > neutral_score:
                    emoji_scores.append(positive_score / 100)  # Positiv skaliert
                elif negative_score > positive_score and negative_score > neutral_score:
                    emoji_scores.append(-negative_score / 100)  # Negativ skaliert
                else:
                    emoji_scores.append(0)  # Neutral
            else:
                emoji_scores.append(0)  # Unbekanntes Emoji ist neutral

        # Durchschnittliches Emoji-Sentiment berechnen
        emoji_sentiment = sum(emoji_scores) / len(emoji_scores)

        # Kombiniertes Sentiment
        return text_sentiment + emoji_sentiment

    # Emojis aus Kommentaren extrahieren
    df["emoji"] = df["comment"].apply(lambda x: "".join([c for c in x if c in emoji_sentiment_data["Emoji"].values]))

    # Kombiniertes Sentiment berechnen
    df["combined_sentiment"] = df.apply(
        lambda row: combined_sentiment(row["comment"], row["emoji"]), axis=1
    )

    # 3. Radar-Diagramm: Cluster-Verteilung
    cluster_counts = df['cluster'].value_counts().sort_index()
    labels = [f"Cluster {i}" for i in cluster_counts.index]
    values = cluster_counts.values
    values = np.append(values, values[0])  # Kreis schließen
    angles = np.linspace(0, 2 * np.pi, len(labels), endpoint=False).tolist()
    angles += angles[:1]

    # Streamlit Radar-Diagramm
    st.subheader("Cluster-Verteilung (Radar-Diagramm)")
    fig, ax = plt.subplots(figsize=(6, 6), subplot_kw=dict(polar=True))
    ax.fill(angles, values, color='blue', alpha=0.25)
    ax.plot(angles, values, color='blue', linewidth=2)
    ax.set_yticks(range(1, max(values) + 1))
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels, size=10)
    ax.set_title("Cluster-Verteilung", size=15, y=1.1)
    st.pyplot(fig)

    # 4. Sentiment-Verlauf über die Zeit
    df['date'] = df['published_at'].dt.date
    time_data = df.groupby('date').agg(
        avg_sentiment=('combined_sentiment', 'mean'),
        comment_count=('comment', 'count')
    ).reset_index()

    st.subheader("Sentiment-Verlauf über die Zeit")
    fig, ax1 = plt.subplots(figsize=(10, 6))
    ax1.plot(time_data['date'], time_data['avg_sentiment'], marker='o', label='Durchschnittliches Sentiment')
    ax1.set_ylabel("Durchschnittliches Sentiment (0-1)")
    ax1.set_xlabel("Datum")
    ax1.set_title("Sentiment-Verlauf über die Zeit")
    ax1.legend(loc='upper left')

    ax2 = ax1.twinx()
    ax2.bar(time_data['date'], time_data['comment_count'], alpha=0.5, color='gray', label='Anzahl Kommentare')
    ax2.set_ylabel("Anzahl Kommentare")
    ax2.legend(loc='upper right')
    st.pyplot(fig)

# Streamlit App
st.title("YouTube Sentiment-Analyse und Clustering")
query = st.selectbox(
    "Wählen Sie ein Smartphone-Modell aus:",
    [
        "iPhone 16", "iPhone 15", "iPhone 14", "iPhone 12", 
        "Samsung Galaxy S23", "Samsung Galaxy S22"
    ]
)
max_videos = st.slider("Maximale Anzahl an Videos", 1, 50, 10)
max_comments = st.slider("Maximale Anzahl an Kommentaren pro Video", 10, 200, 50)

if st.button("Daten abrufen und analysieren"):
    with st.spinner("Kommentare werden abgerufen..."):
        df = get_comments(query, max_videos=max_videos, max_comments_per_video=max_comments)
    st.success("Daten erfolgreich abgerufen!")
    st.subheader("Rohdaten")
    st.dataframe(df.head())
    process_and_visualize_comments(df)
