import pandas as pd
import numpy as np
from transformers import pipeline
import matplotlib.pyplot as plt
import streamlit as st

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