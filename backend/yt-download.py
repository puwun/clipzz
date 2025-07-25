import subprocess
from pytubefix import YouTube
from pytubefix.cli import on_progress

url_01 = "https://www.youtube.com/watch?v=9rTFDi7NjSg"

url_02 = "https://www.youtube.com/watch?v=dlXt8CLb-PM"

url_03 = "https://www.youtube.com/watch?v=TndWQw05wsA"

url_04 = "https://www.youtube.com/watch?v=xAt1xcC6qfM"

yt = YouTube(url_03, on_progress_callback=on_progress)
print(f"Title: {yt.title}")

# ys = yt.streams.get_highest_resolution()
# ys.download()


# Get the highest resolution video-only stream
video_stream = yt.streams.filter(adaptive=True, file_extension="mp4", only_video=True).order_by("resolution").desc().first()

# Get the highest quality audio-only stream
audio_stream = yt.streams.filter(adaptive=True, file_extension="mp4", only_audio=True).order_by("abr").desc().first()

# Download video and audio streams
print("Downloading video...")
video_file = video_stream.download(filename="video.mp4")

print("Downloading audio...")
audio_file = audio_stream.download(filename="audio.mp4")

# Merge video and audio using ffmpeg
output_file = "output_7min.mp4"
print("Merging video and audio...")
subprocess.run([
    "ffmpeg", "-i", video_file, "-i", audio_file, "-c:v", "copy", "-c:a", "aac", "-strict", "experimental", output_file
])

print(f"Download and merge complete: {output_file}")


# yt = YouTube(url_03, on_progress_callback=on_progress)
# print(f"Title: {yt.title}")

# video_stream = yt.streams.filter(adaptive=True, file_extension="mp4", only_video=True).order_by("resolution").desc().first()

# print("Downloading video...")
# video_file = video_stream.download(filename="video.mp4")

