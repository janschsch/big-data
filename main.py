import streamlit as st

import youtube
import analyze

st.title("YouTube Sentiment-Analyse und Clustering")
query = st.selectbox(
    "Wählen Sie ein Smartphone-Modell aus:",
    [
        "iPhone 16", "iPhone 15", "iPhone 14", 
        "Samsung Galaxy S23", "Samsung Galaxy S22"
    ]
)
max_videos = st.slider("Maximale Anzahl an Videos", 1, 50, 10)
max_comments_per_video = st.slider("Maximale Anzahl an Kommentaren pro Video", 10, 200, 50)

if st.button("Daten abrufen und analysieren"):

    with st.spinner("Kommentare werden abgerufen..."):
        data = youtube.get_comments(query, max_videos, max_comments_per_video)

    st.success("Daten erfolgreich abgerufen!")
    st.subheader("Rohdaten")
    st.dataframe(data.head())

    analyze.process_and_visualize_comments(data)