from pydub import AudioSegment

Src = "audio.mp3"
dst = "test.wav"

sound = AudioSegment.from_mp3(Src)
sound.export(dst, format="wav")