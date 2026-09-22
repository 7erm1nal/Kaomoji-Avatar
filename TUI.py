import pyaudio
import numpy as np
import queue
import random
import time
from collections import deque, Counter
from avatar_core import extract_formants, load_profile, predict_sound

# -------------------
# Настройки аудио
# -------------------
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 16000
CHUNK = int(RATE * 0.05)
audio_queue = queue.Queue()

# -------------------
# Словари символов и шкалы (перенесено из avatar_mfcc.py)
# -------------------
MOUTH_SYMBOLS = {
    "А": "▽",
    "О": "O",
    "У": "o",
    "И": "‿",
    "-": "_",
}

EYE_SYMBOLS = {
    "default": "・",
    "blink":   "ー",
    "left":    "＜",
    "right":   "＞",
    "happy":   "＾",
    "angry":   ">",
    "sad":     "T",
}

MOUTH_OPEN_SCALE = ["_", "У", "О", "А"]
MOUTH_TARGET = {"-": 0, "У": 1, "О": 2, "А": 3}

# -------------------
# Callback
# -------------------
def audio_callback(in_data, frame_count, time_info, status):
    audio_data = np.frombuffer(in_data, dtype=np.int16)
    audio_queue.put(audio_data)
    return (None, pyaudio.paContinue)

def face_constructor(state):
    mouth = MOUTH_SYMBOLS.get(state["mouth"], "_")
    eye_l = EYE_SYMBOLS.get(state["eye_left"], "・")
    eye_r = EYE_SYMBOLS.get(state["eye_right"], "・")
    return f"({eye_l} {mouth} {eye_r})"

# -------------------
# Аниматор глаз (независимые таймеры моргания и взгляда)
# -------------------
EYE_BLINK_INTERVAL = (3, 7)
EYE_BLINK_DURATION = (0.1, 0.15)
EYE_LOOK_INTERVAL = (8, 15)
EYE_LOOK_DURATION = (0.5, 1.5)

blink_start = time.time()
blink_next = random.uniform(*EYE_BLINK_INTERVAL)
is_blinking = False

look_start = time.time()
look_next = random.uniform(*EYE_LOOK_INTERVAL)
is_looking = False
look_direction = "left"

def update_eyes(state):
    global blink_start, blink_next, is_blinking
    global look_start, look_next, is_looking, look_direction
    now = time.time()

    if is_blinking:
        if now - blink_start > random.uniform(*EYE_BLINK_DURATION):
            is_blinking = False
            blink_start = now
            blink_next = random.uniform(*EYE_BLINK_INTERVAL)
    else:
        if now - blink_start > blink_next:
            is_blinking = True
            blink_start = now

    if is_looking:
        if now - look_start > random.uniform(*EYE_LOOK_DURATION):
            is_looking = False
            look_start = now
            look_next = random.uniform(*EYE_LOOK_INTERVAL)
    else:
        if now - look_start > look_next:
            is_looking = True
            look_start = now
            look_direction = random.choice(["left", "right"])

    if is_blinking:
        state["eye_left"] = "blink"
        state["eye_right"] = "blink"
    elif is_looking:
        state["eye_left"] = look_direction
        state["eye_right"] = look_direction
    else:
        state["eye_left"] = "default"
        state["eye_right"] = "default"

def update_mouth(state, target_sound):
    target_index = MOUTH_TARGET.get(target_sound, 0)
    current_index = state["mouth_index"]

    if current_index < target_index:
        current_index += 1
    elif current_index > target_index:
        current_index -= 1

    state["mouth_index"] = current_index
    state["mouth"] = MOUTH_OPEN_SCALE[current_index]

# -------------------
# PyAudio init
# -------------------
p = pyaudio.PyAudio()
stream = p.open(format=FORMAT,
                channels=CHANNELS,
                rate=RATE,
                input=True,
                frames_per_buffer=CHUNK,
                stream_callback=audio_callback)

MODEL, SCALER, SILENCE_THRESHOLD = load_profile("mic_profile.joblib")
LEN_HISTORY = 3

avatar_state = {
    "mouth": "-",
    "mouth_index": 0,
    "eye_left": "default",
    "eye_right": "default",
    "emotion": "neutral",
}

history = deque(maxlen=LEN_HISTORY)

print("🎤 Захват начался... (Ctrl+C для выхода)")
stream.start_stream()

# -------------------
# Основной цикл
# -------------------
try:
    while stream.is_active():
        try:
            audio_data = audio_queue.get(timeout=0.05)
            sound = predict_sound(audio_data, MODEL, SCALER, SILENCE_THRESHOLD, RATE)
            history.append(sound)

            most_common_sound = Counter(history).most_common(1)[0][0]

            update_eyes(avatar_state)

            if most_common_sound == "И":
                update_mouth(avatar_state, "У")
                avatar_state["mouth"] = "И"
            else:
                update_mouth(avatar_state, most_common_sound)

            face = face_constructor(avatar_state)
            print(f"\r{face}", end="", flush=True)

        except queue.Empty:
            pass

except KeyboardInterrupt:
    print("\n⏹ Захват остановлен.")

finally:
    stream.stop_stream()
    stream.close()
    p.terminate()