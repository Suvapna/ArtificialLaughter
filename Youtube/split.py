from moviepy.video.io.ffmpeg_tools import ffmpeg_extract_subclip
import soundfile as sf

"""
print('samples = {}'.format(f.frames))
print('sample rate = {}'.format(f.samplerate))

"""
file = sf.SoundFile('test2.wav')

if int(float(format(file.frames / file.samplerate))) < 300:
    print('seconds = {}'.format(file.frames / file.samplerate))

elif int(float(format(file.frames / file.samplerate))) > 300:
    print('seconds = {}'.format(file.frames / file.samplerate))
    # Replace the filename below.
    required_video_file = "test2.wav"

    with open("times.txt") as f:
     times = f.readlines()

    times = [x.strip() for x in times] 

    for time in times:
     starttime = int(time.split("-")[0])
     endtime = int(time.split("-")[1])
     ffmpeg_extract_subclip(required_video_file, starttime, endtime, targetname=str(times.index(time)+1)+".wav")

