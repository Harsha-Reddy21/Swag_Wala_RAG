from googleapiclient.discovery import build
from datetime import datetime, timedelta
import pytz
import isodate  # for parsing ISO 8601 duration
import json
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Get API key from environment variable
API_KEY = os.getenv("YOUTUBE_API_KEY")


def get_channel_videos(channel_name):
    youtube = build('youtube', 'v3', developerKey=API_KEY)

    # Step 1: Get channel ID from channel name
    search_response = youtube.search().list(
        q=channel_name,
        type="channel",
        part="id",
        maxResults=1
    ).execute()

    if not search_response["items"]:
        print("❌ Channel not found.")
        return []

    channel_id = search_response["items"][0]["id"]["channelId"]

    # Step 2: Get Uploads Playlist ID
    channel_response = youtube.channels().list(
        part="contentDetails",
        id=channel_id
    ).execute()

    uploads_playlist_id = channel_response["items"][0]["contentDetails"]["relatedPlaylists"]["uploads"]

    # Step 3: Get all videos
    videos = []
    next_page_token = None

    while True:
        playlist_response = youtube.playlistItems().list(
            part="snippet",
            playlistId=uploads_playlist_id,
            maxResults=50,
            pageToken=next_page_token
        ).execute()

        video_ids = [item["snippet"]["resourceId"]["videoId"] for item in playlist_response["items"]]

        # Get details (duration) for all videos in this batch
        video_details = youtube.videos().list(
            part="contentDetails",
            id=",".join(video_ids)
        ).execute()

        details_map = {item["id"]: item["contentDetails"] for item in video_details["items"]}

        for item in playlist_response["items"]:
            video_id = item["snippet"]["resourceId"]["videoId"]
            title = item["snippet"]["title"]
            published_at_utc = item["snippet"]["publishedAt"]

            # Convert UTC → IST (date only)
            utc_time = datetime.strptime(published_at_utc, "%Y-%m-%dT%H:%M:%SZ")
            utc_zone = pytz.utc
            ist_zone = pytz.timezone("Asia/Kolkata")
            utc_time = utc_zone.localize(utc_time)
            ist_time = utc_time.astimezone(ist_zone)
            upload_date = ist_time.strftime("%Y-%m-%d")

            # Convert ISO 8601 duration → hh:mm:ss
            duration_iso = details_map[video_id]["duration"]
            duration_seconds = int(isodate.parse_duration(duration_iso).total_seconds())
            video_duration = str(timedelta(seconds=duration_seconds))

            videos.append({
                "video_id": video_id,
                "video_url": f"https://www.youtube.com/watch?v={video_id}",
                "video_title": title,
                "upload_date": upload_date,
                "video_duration": video_duration
            })

        next_page_token = playlist_response.get("nextPageToken")
        if not next_page_token:
            break

    return videos


if __name__ == "__main__":
    channel_name = input("Enter YouTube channel name: ")
    video_data = get_channel_videos(channel_name)

    # Save results into a JSON file
    with open("videos.json", "w", encoding="utf-8") as f:
        json.dump(video_data, f, indent=4, ensure_ascii=False)

    print(f"\n✅ Saved {len(video_data)} videos to videos.json")
