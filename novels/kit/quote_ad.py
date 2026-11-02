"""Render one <15s quote ad: quote text synced to voiceover, then cover + NOW ON AMAZON, with suspense music.
usage: python quote_ad.py COVER_PNG VO_MP3 OUT_MP4 "quote with *gold* words" [preview]
"""
import sys, os, subprocess, re, math, wave
import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageFilter

COVER, VO, OUT, QUOTE = sys.argv[1:5]
PREVIEW = len(sys.argv) > 5
W, H, FPS, SR = 1080, 1920, 30, 44100
GOLD = (222, 168, 78); CREAM = (244, 238, 226); GREY = (176, 182, 188)
LORA = "/usr/share/fonts/truetype/google-fonts/Lora-Variable.ttf"
LORA_I = "/usr/share/fonts/truetype/google-fonts/Lora-Italic-Variable.ttf"
INTER_B = "/usr/share/fonts/opentype/inter/Inter-Bold.otf"
INTER_SB = "/usr/share/fonts/opentype/inter/Inter-SemiBold.otf"
INTER_M = "/usr/share/fonts/opentype/inter/Inter-Medium.otf"

def F(p, s, w=None):
    f = ImageFont.truetype(p, s)
    if w:
        try: f.set_variation_by_axes([w])
        except Exception: pass
    return f
def ease(t): t = max(0., min(1., t)); return t * t * (3 - 2 * t)

# ---------- voiceover timing ----------
def decode(path):
    raw = subprocess.run(["ffmpeg", "-v", "error", "-i", path, "-f", "s16le", "-ac", "1", "-ar", str(SR), "-"],
                         capture_output=True, check=True).stdout
    return np.frombuffer(raw, np.int16).astype(np.float32) / 32768
vo = decode(VO)
def _speech(v):
    h = SR // 50
    r = np.sqrt(np.convolve(v ** 2, np.ones(h) / h, "same")[::h])
    return r > 10 ** (-38 / 20)
def _gaps(sp):
    g = []; i = 0; n = len(sp)
    while i < n:
        if not sp[i]:
            j = i
            while j < n and not sp[j]: j += 1
            if (j - i) * 0.02 >= 0.35 and i > 0 and j < n: g.append((i * 0.02, j * 0.02))
            i = j
        else: i += 1
    return g
QUOTE_ONLY = os.environ.get("QUOTE_ONLY") == "1"   # no spoken end line: card appears after the last word
CUT_TAIL = os.environ.get("CUT_TAIL") == "1"       # voice has a spoken end line after a long pause: cut it off
_sp = _speech(vo); _g = _gaps(_sp)
if _g:
    a_, b_ = max(_g, key=lambda g: g[1] - g[0])
    if CUT_TAIL:        # drop the spoken "Title. Now on Amazon." entirely
        vo = vo[:int((a_ + 0.3) * SR)]
    elif b_ - a_ > 1.0:   # shorten the long pause before the title line to 1.0s
        cut0 = int((a_ + 0.5) * SR); cut1 = int((b_ - 0.5) * SR)
        vo = np.concatenate([vo[:cut0], vo[cut1:]])
_sp = _speech(vo)
_last = (len(_sp) - np.argmax(_sp[::-1])) * 0.02
vo = vo[:int(min(len(vo) / SR, _last + 0.35) * SR)]   # trim trailing silence
vo_dur = len(vo) / SR
# find speech segments via 20ms RMS
hop = SR // 50
rms = np.sqrt(np.convolve(vo ** 2, np.ones(hop) / hop, "same")[::hop])
thr = 10 ** (-38 / 20)
speech = rms > thr
# silences longer than 0.35s
gaps = []; i = 0; n = len(speech)
while i < n:
    if not speech[i]:
        j = i
        while j < n and not speech[j]: j += 1
        if (j - i) * 0.02 >= 0.35 and i > 0 and j < n: gaps.append((i * 0.02, j * 0.02))
        i = j
    else: i += 1
first = np.argmax(speech) * 0.02
last = (n - np.argmax(speech[::-1])) * 0.02
LEAD = 0.7
if QUOTE_ONLY or CUT_TAIL:
    quote_end = LEAD + last
    card_t = quote_end + 0.5
    TOTAL = min(14.8, card_t + 3.2)
else:
    longest = max(gaps, key=lambda g: g[1] - g[0])          # the pause before "Title. Now on Amazon."
    quote_end = LEAD + longest[0]
    card_t = LEAD + longest[0] + 0.25
    TOTAL = min(14.8, LEAD + vo_dur + 1.5)

# ---------- quote layout ----------
words = QUOTE.split()
fq = F(LORA, 74, 500); fqi = F(LORA_I, 74, 500)
maxw = W - 2 * 75
lines = []; cur = []
for w in words:
    if w == "/":
        lines.append(cur); cur = []; continue
    test = " ".join(x.strip("*") for x in cur + [w])
    if cur and fq.getlength(test) > maxw: lines.append(cur); cur = [w]
    else: cur.append(w)
lines.append(cur)
words = [w for w in words if w != "/"]
LH = 104
block_h = LH * len(lines)
y0 = (H - block_h) // 2 - 80
# word reveal times: spread over speech before the long pause, weighted by length
speak_s, speak_e = LEAD + first, quote_end
wl = np.array([len(w.strip("*")) + 2 for w in words], float)
starts = speak_s + (np.cumsum(wl) - wl) / wl.sum() * (speak_e - speak_s)

cov = Image.open(COVER).convert("RGB")
def make_bg(cov, dark):
    b = cov.resize((W // 10, H // 10)).filter(ImageFilter.GaussianBlur(5)).resize((W, H), Image.BICUBIC)
    a = np.asarray(b).astype(np.float32) * dark
    yy = np.linspace(-1, 1, H)[:, None]; xx = np.linspace(-1, 1, W)[None, :]
    a *= (1 - 0.6 * np.clip(xx ** 2 * .5 + yy ** 2 * .8, 0, 1))[..., None]
    return Image.fromarray(a.clip(0, 255).astype(np.uint8)).convert("RGBA")
bg_q = make_bg(cov, 0.22); bg_c = make_bg(cov, 0.38)

# grain frames
rng = np.random.default_rng(1)
grain = [Image.fromarray((rng.normal(0, 7, (H // 2, W // 2)) + 128).clip(0, 255).astype(np.uint8)).resize((W, H)) for _ in range(6)]

fsmall = F(INTER_SB, 28); fbig = F(INTER_B, 112); fsub = F(INTER_M, 32); fby = F(LORA_I, 46, 500)
CW = 560
ch = round(cov.height * CW / cov.width)

def tracked(d, x, y, s, f, fill, tr):
    for c in s: d.text((x, y), c, font=f, fill=fill); x += f.getlength(c) + tr
def tw(f, s, tr): return sum(f.getlength(c) for c in s) + tr * (len(s) - 1)

def frame(t):
    if t < card_t + 0.5:
        img = bg_q.copy()
        L = Image.new("RGBA", (W, H), (0, 0, 0, 0)); d = ImageDraw.Draw(L)
        a = ease((t - 0.1) / 0.5)
        s = "DESMOND KEARWOOD"; tracked(d, (W - tw(fsmall, s, 9)) / 2, 230, s, fsmall, GOLD + (int(200 * a),), 9)
        # big faint quote mark
        d.text((100, y0 - 170), "“", font=F(LORA, 260, 500), fill=GOLD + (int(70 * a),))
        k = 0
        for li, ln in enumerate(lines):
            txt = [w.strip("*") for w in ln]
            lw = fq.getlength(" ".join(txt)); x = (W - lw) / 2; y = y0 + li * LH
            for w in ln:
                gold = w.startswith("*") or w.endswith("*")
                ww = w.strip("*")
                p = ease((t - starts[k]) / 0.45)
                f = fqi if gold else fq
                if p > 0:
                    d.text((x, y + (1 - p) * 14), ww, font=f, fill=(GOLD if gold else CREAM) + (int(255 * p),))
                x += (fqi if gold else fq).getlength(ww) + fq.getlength(" "); k += 1
        img.alpha_composite(L)
        out = img
        if t > card_t:   # crossfade to card
            mix = ease((t - card_t) / 0.5)
            out = Image.blend(img, card(t), mix)
    else:
        out = card(t)
    rgb = out.convert("RGB")
    g = grain[int(t * FPS) % len(grain)]
    rgb = Image.blend(rgb, Image.merge("RGB", (g, g, g)), 0.05)
    fade = min(ease(t / 0.35), ease((TOTAL - t) / 0.4))
    if fade < 1: rgb = Image.fromarray((np.asarray(rgb).astype(np.float32) * fade).astype(np.uint8))
    return rgb

def card(t):
    tt = t - card_t
    img = bg_c.copy()
    z = 1.0 + 0.04 * max(0, tt) / 4
    w = round(CW * z); h = round(ch * z)
    c = cov.resize((w, h), Image.LANCZOS)
    pad = 70
    sh = Image.new("RGBA", (w + 2 * pad, h + 2 * pad), (0, 0, 0, 0))
    ImageDraw.Draw(sh).rectangle([pad + 8, pad + 22, pad + w + 8, pad + h + 22], fill=(0, 0, 0, 200))
    sh = sh.filter(ImageFilter.GaussianBlur(28)); sh.paste(c, (pad, pad))
    ImageDraw.Draw(sh).line([(pad + 5, pad), (pad + 5, pad + h)], fill=(255, 255, 255, 45), width=3)
    img.alpha_composite(sh, ((W - sh.width) // 2, 200 - pad - (h - ch) // 2))
    L = Image.new("RGBA", (W, H), (0, 0, 0, 0)); d = ImageDraw.Draw(L)
    a = ease((tt - 0.35) / 0.45)
    for s, y, col in [("NOW ON", 1120, CREAM), ("AMAZON", 1235, GOLD)]:
        tracked(d, (W - tw(fbig, s, 6)) / 2, y + (1 - a) * 25, s, fbig, col + (int(255 * a),), 6)
    a2 = ease((tt - 0.8) / 0.45)
    s = "KINDLE  ·  PAPERBACK  ·  HARDCOVER"
    tracked(d, (W - tw(fsub, s, 5)) / 2, 1395, s, fsub, GREY + (int(255 * a2),), 5)
    a3 = ease((tt - 1.1) / 0.5)
    s = "by Desmond Kearwood"; d.text(((W - fby.getlength(s)) / 2, 1465), s, font=fby, fill=CREAM + (int(255 * a3),))
    img.alpha_composite(L)
    return img

if PREVIEW:
    for name, t in [("q", quote_end - 0.1), ("c", min(TOTAL - 0.6, card_t + 2))]:
        frame(t).save(OUT.replace(".mp4", f"_{name}.jpg"), quality=85)
    print("card_t", card_t, "total", TOTAL); sys.exit()

# ---------- suspense music ----------
N = int(TOTAL * SR); tt = np.arange(N) / SR
def nf(m): return 440 * 2 ** ((m - 69) / 12)
mus = np.zeros(N)
# low drone: D minor-ish, slow swell, detuned
for m, amp in [(38, .22), (45, .12), (50, .08)]:
    f = nf(m)
    mus += amp * (np.sin(2 * np.pi * f * tt) + .6 * np.sin(2 * np.pi * f * 1.004 * tt + 1) + .25 * np.sin(2 * np.pi * 2 * f * tt))
mus *= np.minimum(1, tt / 2.0)
# eerie high tremolo string (minor second rub), creeping in
hi = (np.sin(2 * np.pi * nf(81) * tt + 3 * np.sin(2 * np.pi * 5.5 * tt) * .02) + .7 * np.sin(2 * np.pi * nf(82) * tt))
hi *= (0.5 + 0.5 * np.sin(2 * np.pi * 7 * tt)) * np.clip((tt - 1.5) / 6, 0, 1) * .05
mus += hi
# heartbeat pulse, speeding up slightly until the card
beat = np.zeros(N); bt = 0.6; period = 1.0
while bt < card_t:
    for off, a in [(0, 1), (0.16, .6)]:
        s0 = int((bt + off) * SR)
        if s0 < N:
            k = np.arange(min(int(.35 * SR), N - s0)) / SR
            beat[s0:s0 + len(k)] += a * np.exp(-k * 18) * np.sin(2 * np.pi * (55 + 40 * np.exp(-k * 30)) * k)
    bt += period; period = max(0.72, period * 0.96)
mus += .55 * beat
# soft clock tick
for k_ in np.arange(0.4, card_t, 0.5):
    s0 = int(k_ * SR); L_ = int(.025 * SR)
    if s0 + L_ < N: mus[s0:s0 + L_] += .06 * np.random.default_rng(int(k_ * 10)).normal(0, 1, L_) * np.exp(-np.arange(L_) / SR * 220)
# riser into the card + impact
r0 = max(0, card_t - 1.6); m_ = (tt >= r0) & (tt < card_t)
x = tt[m_] - r0
mus[m_] += .10 * (x / 1.6) ** 2 * np.random.default_rng(3).normal(0, 1, m_.sum())
m_ = tt >= card_t; x = tt[m_] - card_t
mus[m_] += .9 * np.exp(-x * 2.5) * np.sin(2 * np.pi * (42 + 30 * np.exp(-x * 6)) * x)
mus[m_] += .25 * np.exp(-x * 1.2) * np.random.default_rng(4).normal(0, 1, m_.sum()) * np.exp(-x * 6)
# simple reverb
ir_n = int(1.2 * SR); ir = np.random.default_rng(5).normal(0, 1, ir_n) * np.exp(-np.arange(ir_n) / SR * 4); ir[0] = 6
mus = np.convolve(mus, ir)[:N] / 8
mus *= np.minimum(1, (TOTAL - tt) / 0.8)
mus /= np.abs(mus).max() + 1e-9
# place voice, duck music under speech
voice = np.zeros(N); s0 = int(LEAD * SR); voice[s0:s0 + len(vo)] = vo[:N - s0]
env = np.convolve(np.abs(voice), np.ones(int(.15 * SR)) / (.15 * SR), "same")
duck = 1 - 0.55 * np.clip(env / (env.max() * .25 + 1e-9), 0, 1)
mix = 0.30 * mus * duck + 1.0 * voice / (np.abs(voice).max() + 1e-9) * 0.85
mix /= max(1, np.abs(mix).max() / 0.95)
wav = OUT.replace(".mp4", ".wav")
with wave.open(wav, "wb") as w:
    w.setnchannels(1); w.setsampwidth(2); w.setframerate(SR); w.writeframes((mix * 32767).astype(np.int16).tobytes())

tmp = OUT.replace(".mp4", ".tmp.mp4")
p = subprocess.Popen(["ffmpeg", "-y", "-v", "error", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{W}x{H}", "-r", str(FPS), "-i", "-",
                      "-i", wav, "-map", "0:v", "-map", "1:a", "-c:v", "libx264", "-preset", "veryfast", "-crf", "19", "-pix_fmt", "yuv420p",
                      "-c:a", "aac", "-b:a", "192k", "-ac", "2", "-shortest", "-movflags", "+faststart", tmp], stdin=subprocess.PIPE)
for i in range(round(TOTAL * FPS)): p.stdin.write(frame(i / FPS).tobytes())
p.stdin.close()
if p.wait() != 0: sys.exit("ffmpeg failed")
os.replace(tmp, OUT); print("ok", OUT, round(TOTAL, 2))
