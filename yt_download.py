import os
import yt_dlp

def download_youtube_video(youtube_url, save_path='data\\Youtube_files'):
    try:
        if not os.path.exists(save_path):
            os.makedirs(save_path)

        # Clean the URL (remove playlist/radio parameters)
        clean_url = youtube_url.split('&')[0]

        ydl_opts = {
            'outtmpl': os.path.join(save_path, '%(title)s.%(ext)s'),
            'format': 'best'
            # Don't add 'noplaylist': True here since the URL is already cleaned
        }

        print("Fetching video information...")
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            ydl.download([clean_url])

        print(f"\n✅ Download completed! Video saved to: {save_path}")

    except Exception as e:
        print(f"❌ An error occurred: {e}")

if __name__ == "__main__":
    youtube_link = input("Enter the YouTube video link: ").strip()
    download_youtube_video(youtube_link)
