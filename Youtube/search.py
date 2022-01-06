import urllib.request
import re

search_keyword = "%23laugh" 
html = urllib.request.urlopen("https://www.youtube.com/results?search_query=" + search_keyword)
video_ids = re.findall(r"watch\?v=(\S{11})",html.read().decode())

x = 0
for video_id in video_ids:  
    print("https://www.youtube.com/watch?v=" + video_ids[x])
    x += 1
    