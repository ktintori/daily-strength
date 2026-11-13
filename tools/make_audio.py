"""Makes a natural-sounding MP3 for every message in messages.json that doesn't have one yet.
Voices: Kokoro af_heart for English, ef_dora for Spanish (open source, free). Run: python tools/make_audio.py <model_dir>"""
import json, os, subprocess, sys
import numpy as np, soundfile as sf
from kokoro_onnx import Kokoro

model_dir = sys.argv[1]
root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(root)
def load(name):
    return np.fromfile(os.path.join(model_dir, name + ".bin"), dtype=np.float32).reshape(-1, 1, 256)
npz = os.path.join(model_dir, "voices.npz")
np.savez(npz, af_heart=load("af_heart"), ef_dora=load("ef_dora"))
k = Kokoro(os.path.join(model_dir, "model.onnx"), npz)

messages = json.load(open("messages.json"))
os.makedirs("audio", exist_ok=True)
made = 0
def record(out, parts, voice, code):
    pieces, sr = [], 24000
    for p in parts:
        smp, sr = k.create(p, voice=voice, speed=0.9, lang=code)
        pieces += [smp, np.zeros(int(sr * 0.6), dtype=np.float32)]
    wav = out[:-4] + ".wav"
    sf.write(wav, np.concatenate(pieces), sr)
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", wav, "-ac", "1", "-b:a", "64k", out], check=True)
    os.remove(wav)
    print("made", out)

for m in messages:
    out = f"audio/{m['date']}.mp3"
    if not os.path.exists(out):
        parts = [m["focus"] + ".", m["message"]]
        if m.get("practice"):
            parts += ["Try this today.", m["practice"]]
        record(out, parts, "af_heart", "en-us")
        made += 1
    es = m.get("es")
    out = f"audio/{m['date']}-es.mp3"
    if es and not os.path.exists(out):
        parts = [es["focus"] + ".", es["message"]]
        if es.get("practice"):
            parts += ["Haz esto hoy.", es["practice"]]
        record(out, parts, "ef_dora", "es")
        made += 1
print(made, "new audio files")
