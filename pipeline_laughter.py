import os, sys, pickle, time, librosa, argparse, torch, numpy as np, pandas as pd, scipy
from tqdm import tqdm
import tgt
sys.path.append('./utils/')
import laugh_segmenter
import models, configs
import dataset_utils, audio_utils, data_loaders, torch_utils
from tqdm import tqdm
from torch import optim, nn
from functools import partial
from distutils.util import strtobool
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
import shutil
import glob

output_dir = "./laughter"

def createCsv():
    #create new csv
    if not os.path.isfile('./results.csv'):
        with open('results.csv', 'a', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(["Hashtags,Link, Title", "Laughter"])
    elif os.path.isfile('./results.csv'):
        print("File already exist")

def search():
    with open('keywords.csv', 'r') as read_obj:
        csv_reader = reader(read_obj)
        for row in csv_reader:

            hashtag = str(row[0])
            #search yt-video with hastags
            search_keyword = ("%23"+str(row[0]))
            html = urllib.request.urlopen("https://www.youtube.com/results?search_query=" + search_keyword)
            video_ids = re.findall(r"watch\?v=(\S{11})",html.read().decode())

            x = 0
            #concat all found video ids with link
            for video_id in video_ids:

                video = ("https://www.youtube.com/watch?v=" + video_ids[x])

                print (row[0])
                print("https://www.youtube.com/watch?v=" + video_ids[x])

                #call download function

                try:
                    download(video,hashtag)
                except Exception:
                    print("Failed to Download!: " + video)
                pass


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

            #change filename to convert
            if(filename[-3:]=="m4a"):
                newFile = filename[:-4]
                #print("new Filename of m4a is: " + newFile)
            elif(filename[-3:]=="ebm"):
                newFile = filename[:-5]
                #print("new Filename of webm is: " + newFile)

            moveFile()
            convert(newFile)
            split(newFile, video, video_title,hashtag)



def convert(audio):
        src = audio + ".mp3"
        dst = audio + ".wav"

        print("Audiofile:" + audio)

        #convert from mp3 to wav
        sound = AudioSegment.from_mp3(src)
        sound.export(dst, format="wav")

        #delete mp3
        os.remove(audio + ".mp3")



def split(audio, video, video_title, hashtag):
    file = sf.SoundFile(audio +'.wav')

    #if wav less than 5 min do nothing --> CHANGE TO <
    if int(float(format(file.frames / file.samplerate))) < 300:
        print('seconds = {}'.format(file.frames / file.samplerate))

        #get the laugh ( audio or newFile?)

        try:
            segment_laughter(audio, output_dir, hashtag, video, video_title)
        except Exception as x:
            print(f"Skip Exception!: {audio} ({x})")
        pass

        #print("Final File : " + audio)

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
            #create subclib
            ffmpeg_extract_subclip(required_video_file, starttime, endtime, targetname=str(times.index(time)+1)+ "-" + audio +".wav")

            n += 1

            #get the laugh
            print("Final File Part : " + str(times.index(time)+1) + "-" + audio)
            #try catch block- if file too small
            try:
                segment_laughter(str(times.index(time)+1) + "-" + audio,
                                 output_dir, hashtag, video, video_title)
            except Exception as x:
                print(f"Skip Exception!: {str(times.index(time)+1)} - {audio} ({x})")
            pass




def segment_laughter(wav_file, output_dir, hashtag, video, video_title):
    sample_rate = 8000

    model_path = "checkpoints/in_use/resnet_with_augmentation"
    config = configs.CONFIG_MAP["resnet_with_augmentation"]
    audio_path = wav_file +".wav"
    threshold = float(0.5)
    min_length = float(0.2)
    save_to_audio_files = bool(strtobool("True"))
    save_to_textgrid = bool(strtobool("False"))

    # represents the data type of a torch( cpu/cuda)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    #print(f"Using device {device}")

    # Load the Model

    model = config['model'](dropout_rate=0.0, linear_layer_size=config['linear_layer_size'], filter_sizes=config['filter_sizes'])
    feature_fn = config['feature_fn']
    model.set_device(device)

    if os.path.exists(model_path):
        torch_utils.load_checkpoint(model_path+'/best.pth.tar', model)
        model.eval()
    else:
        raise Exception(f"Model checkpoint not found at {model_path}")

    # Load the audio file and features

    inference_dataset = data_loaders.SwitchBoardLaughterInferenceDataset(
        audio_path=audio_path, feature_fn=feature_fn, sr=sample_rate)

    collate_fn=partial(audio_utils.pad_sequences_with_labels,
                            expand_channel_dim=config['expand_channel_dim'])

    inference_generator = torch.utils.data.DataLoader(
        inference_dataset, num_workers=0, batch_size=8, shuffle=False, collate_fn=collate_fn)


    # Make Predictions

    probs = []
    for model_inputs, _ in tqdm(inference_generator):
        x = torch.from_numpy(model_inputs).float().to(device)
        preds = model(x).cpu().detach().numpy().squeeze()
        if len(preds.shape)==0:
            preds = [float(preds)]
        else:
            preds = list(preds)
        probs += preds
    probs = np.array(probs)

    file_length = audio_utils.get_audio_length(audio_path)

    fps = len(probs)/float(file_length)

    probs = laugh_segmenter.lowpass(probs)
    instances = laugh_segmenter.get_laughter_instances(probs, threshold=threshold, min_length=min_length, fps=fps)

    print(); print("found %d laughs." % (len (instances)))

    #if no laughter found
    if len(instances) == 0:
        with open('results.csv', 'a', newline='',encoding="utf-8") as file:
                        writer = csv.writer(file)
                        writer.writerow([hashtag,video,wav_file,"found 0 laughs"])
    #if laughter found
    elif len(instances) > 0:
        full_res_y, full_res_sr = librosa.load(audio_path,sr=44100)
        wav_paths = []
        maxv = np.iinfo(np.int16).max

        if save_to_audio_files:
            if output_dir is None:
                raise Exception("Need to specify an output directory to save audio files")
            else:
                os.system(f"mkdir -p {output_dir}")
                for index, instance in enumerate(instances):
                    laughs = laugh_segmenter.cut_laughter_segments([instance],full_res_y,full_res_sr)
                    wav_path = output_dir + "/"+ wav_file + str(index) + ".wav"
                    scipy.io.wavfile.write(wav_path, full_res_sr, (laughs * maxv).astype(np.int16))
                    wav_paths.append(wav_path)

                with open('results.csv', 'a', newline='',encoding="utf-8") as file:
                        writer = csv.writer(file)
                        writer.writerow([hashtag,video,wav_file,laugh_segmenter.format_outputs(instances, wav_paths)])

                        print(laugh_segmenter.format_outputs(instances, wav_paths))


        if save_to_textgrid:
            laughs = [{'start': i[0], 'end': i[1]} for i in instances]
            tg = tgt.TextGrid()
            laughs_tier = tgt.IntervalTier(name='laughter', objects=[
            tgt.Interval(l['start'], l['end'], 'laugh') for l in laughs])
            tg.add_tier(laughs_tier)
            fname = os.path.splitext(os.path.basename(audio_path))[0]
            tgt.write_to_file(tg, os.path.join(output_dir, fname + '_laughter.TextGrid'))

            print('Saved laughter segments in {}'.format(
                os.path.join(output_dir, fname + '_laughter.TextGrid')))


def moveFile():
    try:
        for data in glob.glob(r"C:.\*.wav"):
            shutil.move(data,r"C:.\audio")
    except Exception:
        print("Failed move the File")
    pass


createCsv()
search()
