import urllib.request
import re
from csv import reader
from csv import writer
import csv
import youtube_dl
import sys
import pandas as pd
from pydub import AudioSegment
from moviepy.video.io.ffmpeg_tools import ffmpeg_extract_subclip
import soundfile as sf


def createCsv():
    #create new csv
    with open('results.csv', 'a', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(["Hashtags,Link, Title"])

def search():
    with open('keywords.csv', 'r') as read_obj:
            csv_reader = reader(read_obj)
            for row in csv_reader:
                #save Hashtags in csv
                #with open('results.csv', 'a', newline='') as file:
                        #writer = csv.writer(file)
                        #writer.writerow([str(row[0])])

                hashtag = str(row[0])
                #search yt-video with hastags
                search_keyword = ("%23"+str(row[0]))
                html = urllib.request.urlopen("https://www.youtube.com/results?search_query=" + search_keyword)
                video_ids = re.findall(r"watch\?v=(\S{11})",html.read().decode())

                x = 0
                #concat all found video ids with link
                for video_id in video_ids:

                    video = ("https://www.youtube.com/watch?v=" + video_ids[x])

                    #new Column with yt-videos
                    #df = pd.read_csv("results.csv")
                    #df["Youtube Link"] = video
                    #df.to_csv("results.csv", index=False)

                    print (row[0])
                    print("https://www.youtube.com/watch?v=" + video_ids[x])
                    
                    #call download function
                    download(video,hashtag)
                    x += 1


def download(video,hashtag):
    ydl_opts = {
        'format': 'bestaudio/best',
        'postprocessors': [{
            'key': 'FFmpegExtractAudio',
            'preferredcodec': 'mp3',
            'preferredquality': '192',
        }],
    }

    if __name__ == "__main__":
        with youtube_dl.YoutubeDL(ydl_opts) as ydl:
            
            #get video informations
            info_dict = ydl.extract_info(video, download=False)
            video_url = info_dict.get("url", None)
            video_id = info_dict.get("id", None)
            video_title = info_dict.get('title', None)
            filename = ydl.prepare_filename(info_dict)
            print("Titel: " + video_title)
            print(video_id)
            print("Filename: " + filename)
           
    
            #download the video
            ydl.download([video])
            
            #create new column with title in csv
            #df = pd.read_csv("results.csv")
            #df["Title"] = video_title
            #df.to_csv("results.csv", index=False)

            #change filename to convert
            if(filename[-3:]=="m4a"):
                newFile = filename[:-4]
                print("new Filename of m4a is: " + newFile)
            elif(filename[-3:]=="ebm"):
                newFile = filename[:-5]
                print("new Filename of webm is: " + newFile)

            convert(newFile)
            split(newFile, video, video_title,hashtag)

              


def convert(audio):
        Src = audio + ".mp3"
        dst = audio + ".wav"

        #convert from mp3 to wav
        sound = AudioSegment.from_mp3(Src)
        sound.export(dst, format="wav")


def split(audio, video, video_title, hashtag):
    file = sf.SoundFile(audio +'.wav')

    #if wav less than 5 min do nothing
    if int(float(format(file.frames / file.samplerate))) < 300:
        print('seconds = {}'.format(file.frames / file.samplerate))

        with open('results.csv', 'a', newline='') as file:
                        writer = csv.writer(file)
                        writer.writerow([hashtag,video,video_title])

    # if wav longer than 5 min cut 
    elif int(float(format(file.frames / file.samplerate))) > 300:
        print('seconds = {}'.format(file.frames / file.samplerate))
        # Replace the filename below.
        required_video_file = audio+".wav"

        with open("times.txt") as f:
            times = f.readlines()

        times = [x.strip() for x in times] 

        n = 0
        for time in times:
            starttime = int(time.split("-")[0])
            endtime = int(time.split("-")[1])
            ffmpeg_extract_subclip(required_video_file, starttime, endtime, targetname=str(times.index(time)+1)+ "-" + audio +".wav")

            n += 1
            
            #df.drop(df.tail(1).index,inplace=True) # drop last n rows
            with open('results.csv', 'a', newline='', encoding="utf-8") as file:
                        writer = csv.writer(file)
                        writer.writerow([hashtag,video,video_title + str(n)])
          
                  






createCsv()
search()