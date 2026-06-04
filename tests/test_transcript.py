from faster_whisper import WhisperModel

model = WhisperModel("tiny.en", device="cpu", compute_type="int8")
segments, info = model.transcribe("/home/oosman/Desktop/loud.wav")

print("Language:", info.language)
for segment in segments:
    print(segment.text)