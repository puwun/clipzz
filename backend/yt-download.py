from pytubefix import YouTube
from pytubefix.cli import on_progress

url_01 = "https://www.youtube.com/watch?v=9rTFDi7NjSg"

url_02 = "https://www.youtube.com/watch?v=dlXt8CLb-PM"

yt = YouTube(url_02, on_progress_callback=on_progress)
print(f"Title: {yt.title}")

ys = yt.streams.get_highest_resolution()
ys.download()