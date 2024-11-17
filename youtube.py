import time
import json
import pandas as pd
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError

def get_comments(query, max_videos=100, max_comments_per_video=200):

    API_KEY = 'AIzaSyBtf1FslpfXTpGgCStPMFrkW32M0U9EeLE'

    # Initialize YouTube service
    youtube = build('youtube', 'v3', developerKey=API_KEY)

    # Fetch videos based on the query
    videos = []
    next_page_token = None

    while len(videos) < max_videos:
        request = youtube.search().list(
            q=query,
            part='id,snippet',
            type='video',
            order='viewCount',  # or 'relevance'
            maxResults=50,  # Maximum allowed per request
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

        # To be safe with rate limits
        time.sleep(0.1)

    videos = videos[:max_videos]

    # Fetch comments for each video
    all_comments = []

    for idx, video in enumerate(videos):
        video_id = video['video_id']
        video_title = video['video_title']
        print(f"Processing video {idx+1}/{len(videos)}: {video_id} | {video_title}")
        next_page_token = None
        comments = []

        while len(comments) < max_comments_per_video:
            try:
                request = youtube.commentThreads().list(
                    part='snippet',
                    videoId=video_id,
                    maxResults=100,  # Maximum allowed per request
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

                # To be safe with rate limits
                time.sleep(0.1)

            except HttpError as e:
                error_content = json.loads(e.content.decode('utf-8'))
                error_reason = error_content['error']['errors'][0]['reason']
                if error_reason == 'commentsDisabled':
                    print(f"Comments are disabled for video {video_id}.")
                else:
                    print(f"An HTTP error occurred while fetching comments for video {video_id}: {e}")
                break  # Exit the loop if an error occurs

        all_comments.extend(comments[:max_comments_per_video])

        # To be safe with rate limits and quotas
        time.sleep(0.1)

    # Create DataFrame
    df = pd.DataFrame(all_comments)

    # Keep only the required columns
    df = df[['published_at', 'like_count', 'comment']]

    # Sort DataFrame by like count in descending order
    df_sorted = df.sort_values(by='like_count', ascending=False)

    return df_sorted