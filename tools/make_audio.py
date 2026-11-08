"""Makes a natural-sounding MP3 for every message in messages.json that doesn't have one yet.
Voice: Kokoro af_heart (open source). Run: python tools/make_audio.py <model_dir>"""
import json, os, subprocess, sys
import numpy as np, soundfile as sf
from kokoro_onnx import Kokoro

model_dir = sys.argv[1]
root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(root)
voice = np.fromfile(os.path.join(model_dir, "af_heart.bin"), dtype=np.float32).reshape(-1, 1, 256)
npz = os.path.join(model_dir, "voices.npz")
np.savez(npz, af_heart=voice)
k = Kokoro(os.path.join(model_dir, "model.onnx"), npz)

messages = json.load(open("messages.json"))
os.makedirs("audio", exist_ok=True)
made = 0
for m in messages:
    out = f"audio/{m['date']}.mp3"
    if os.path.exists(out):
        continue
    parts = [m["focus"] + ".", m["message"]]
    if m.get("practice"):
        parts += ["Try this today.", m["practice"]]
    pieces, sr = [], 24000
    for p in parts:
        s, sr = k.create(p, voice="af_heart", speed=0.9, lang="en-us")
        pieces += [s, np.zeros(int(sr * 0.6), dtype=np.float32)]
    wav = out[:-4] + ".wav"
    sf.write(wav, np.concatenate(pieces), sr)
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", wav, "-ac", "1", "-b:a", "64k", out], check=True)
    os.remove(wav)
    made += 1
    print("made", out)
print(made, "new audio files")
