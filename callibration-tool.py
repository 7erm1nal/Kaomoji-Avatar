import pyaudio
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import librosa
from scipy.signal import find_peaks
import joblib
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier


CHUNK = 1024
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 16000
DURATION = 5                        # Продолжительность записи одного звука в секундах
RECORD_SOUNDS=["И", "У", "А", "О", "Э", "М"]  # Массив звуков для записи и калибровки
BLOCK_SIZE = int(RATE * 0.05)       # Блоки для анализа 0.05 -- 50 мс
SILENCE_THRESHOLD = 500             # Порог тишины. от 0 до 32767 для paInt16

def record_sound(p):
    """
    Функция записи звукового фрагмента
    На вход подается экземпляр класса PyAudio
    """
    audio_array = None
    stream = None
    try:
        stream = p.open(format=FORMAT,
                channels=CHANNELS,
                rate=RATE,
                input=True,
                frames_per_buffer=CHUNK)

        K=int(RATE / CHUNK * DURATION)
        audio_data=[]
        for _ in tqdm(range(K), desc="Запись"):
            audio_data.append(np.frombuffer(stream.read(CHUNK), dtype=np.int16))

        audio_array=np.concatenate(audio_data)
    finally:
        if stream is not None:
            stream.stop_stream()
            stream.close()
    
    return audio_array

def debug_lpc_spectrum(block, sound_name):
    """
    Дебаг-функция для вывода спектрограммы одного звука после записи
    На вход: блок звука и название звука
    На выход: график спектрограммы с centroid, ratio и slope
    """
    block = block - np.mean(block)
    block = block.astype(np.float32)
    block_windowed = block * np.hamming(len(block))

    lpc_coeffs = librosa.lpc(block_windowed, order=(2 + RATE // 1000))

    fft_lpc = np.fft.fft(lpc_coeffs, n=8192)
    lpc_spectrum = 1.0 / np.abs(fft_lpc[:4096])

    freqs = np.fft.fftfreq(8192, 1/RATE)[:4096]

    mask = (freqs >= 0) & (freqs <= 3000)
    freqs = freqs[mask]
    lpc_spectrum = lpc_spectrum[mask]

    plt.plot(freqs, lpc_spectrum)
    
    #Спектральный центроид. Центр тяжести спектра — средняя частота взвешенная по амплитуде:
    centroid = np.sum(freqs * lpc_spectrum) / np.sum(lpc_spectrum)

    #Соотношение энергий полос. Энергия в полосе — это сумма квадратов амплитуд в этой полосе
    mask_low = (freqs >= 100) & (freqs < 300)
    mask_mid = (freqs >= 300) & (freqs <= 600)
    energy_low = np.sum(lpc_spectrum[mask_low] ** 2)
    energy_mid = np.sum(lpc_spectrum[mask_mid] ** 2)
    ratio = energy_mid / energy_low

    slope = None

    # F1 — строгий порог, весь диапазон
    peaks_f1, _ = find_peaks(lpc_spectrum, prominence=np.max(lpc_spectrum)*0.1, distance=100)

    if len(peaks_f1) > 0:
        # берём самый сильный пик как F1
        f1_idx = peaks_f1[np.argmax(lpc_spectrum[peaks_f1])]
        plt.scatter(freqs[f1_idx], lpc_spectrum[f1_idx], color='red', zorder=5, label='F1')
        
        f1_freq = freqs[f1_idx]
        # находим индекс ближайшей частоты к f1_freq + 400
        slope_idx = np.argmin(np.abs(freqs - (f1_freq + 400)))
        slope = lpc_spectrum[slope_idx] / lpc_spectrum[f1_idx]
        
        # F2 — мягкий порог, только правее F1
        mask_f2 = freqs > freqs[f1_idx] + 200
        if np.any(mask_f2):
            lpc_f2 = lpc_spectrum.copy()
            lpc_f2[~mask_f2] = 0  # обнуляем всё левее границы
            peaks_f2, properties_f2 = find_peaks(lpc_f2, prominence=np.max(lpc_spectrum)*0.03, distance=100)
            if len(peaks_f2) > 0:
                f2_idx = peaks_f2[np.argmax(properties_f2["prominences"])]
                plt.scatter(freqs[f2_idx], lpc_spectrum[f2_idx], color='blue', zorder=5, label='F2')
    
    #Крутизна спада после пика. После того как найден F1 (индекс f1_idx), берём амплитуды на 200 Гц правее F1 и на 400 Гц правее F1, и смотрим насколько упала амплитуда
    slope_str = f"{slope:.3f}" if slope is not None else "N/A"
    plt.text(0.05, 0.95, f"centroid={centroid:.1f} Hz\nratio={ratio:.3f}\nslope={slope_str}", transform=plt.gca().transAxes, verticalalignment='top')

    plt.title(sound_name)
    plt.xlabel("Частота Гц")
    plt.ylabel("Амплитуда LPC")
    plt.legend()
    plt.show()

def find_formants(freq_from, freq_to, amplitudes, freqs, prom):
    """
    Функция поиска форманты.
    На вход: границы частот поиска, амплитуда, частоты, чувствительность
    """
    mask = (freqs >= freq_from) & (freqs <= freq_to)
    amplitudes_in_range = amplitudes[mask]
    freqs_in_range = freqs[mask]
    
    peaks, properties = find_peaks(amplitudes_in_range, prominence=(np.max(amplitudes_in_range) * prom), distance=100)
    if len(peaks) == 0:
        best_peak = np.argmax(amplitudes_in_range)
    else:
        best_peak = peaks[np.argmax(properties["prominences"])]

    return freqs_in_range[best_peak]

def extract_formants(block):
    """
    Функция извлечения формант F1 и F2 из звукового блока
    На вход подается звуковой блок, на выходе F1, F2, centroid, ratio и RMS этого блока
    """
    block = block.astype(np.float32)
    block_windowed = block * np.hamming(len(block))

    lpc_coeffs = librosa.lpc(block_windowed, order=(2 + RATE // 1000))

    fft_lpc = np.fft.fft(lpc_coeffs, n=8192)
    lpc_spectrum = 1.0 / np.abs(fft_lpc[:4096])

    freqs = np.fft.fftfreq(8192, 1/RATE)[:4096]

    f1 = find_formants(100, 1200, lpc_spectrum, freqs, 0.1)     # Разная чувствительность для выделения пика форманты
    f2 = find_formants(400, 3000, lpc_spectrum, freqs, 0.03)

    mask = (freqs >= 0) & (freqs <= 3000)
    freqs = freqs[mask]
    lpc_spectrum = lpc_spectrum[mask]

    #Спектральный центроид. Центр тяжести спектра — средняя частота взвешенная по амплитуде:
    centroid = np.sum(freqs * lpc_spectrum) / np.sum(lpc_spectrum)

    #Соотношение энергий полос. Энергия в полосе — это сумма квадратов амплитуд в этой полосе
    mask_low = (freqs >= 100) & (freqs < 300)
    mask_mid = (freqs >= 300) & (freqs <= 600)
    energy_low = np.sum(lpc_spectrum[mask_low] ** 2)
    energy_mid = np.sum(lpc_spectrum[mask_mid] ** 2)
    ratio = energy_mid / energy_low

    rms = np.sqrt(np.mean(block**2))

    return f1, f2, centroid, ratio, rms

def process_recording(audio_array, sound_name):
    """
    Нарезает звуковую дорожку на блоки длины BLOCK_SIZE и отправляет в функцию extract_formants
    Возвращает список кортежей вида [(f1, f2, centroid, ratio), (f1, f2, centroid, ratio), ...]
    """
    debug_done = False

    formants = []
    for i in range(0, len(audio_array), BLOCK_SIZE):
        block = audio_array[i : i + BLOCK_SIZE]
        if len(block) < BLOCK_SIZE: continue

        f1, f2, centroid, ratio, rms= extract_formants(block)
        if rms > SILENCE_THRESHOLD:
            formants.append((f1, f2, centroid, ratio))
            if not debug_done:
                debug_lpc_spectrum(block, sound_name)   #Запуск дебаг-функции отрисовки спектрограммы
                debug_done = True

    return formants

def build_dataset(all_points:dict):
    X=[]
    Y=[]
    for key, points in all_points.items():
        for point in points:
            X.append(point)
            Y.append(key)
    
    return np.array(X), np.array(Y)

def train_classifier(X, Y):
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    model = KNeighborsClassifier(n_neighbors=5)
    model.fit(X_scaled, Y)
    
    return model, scaler

def save_profile(model, scaler):
    profile={
        "model": model,
        "scaler": scaler,
        "sounds": RECORD_SOUNDS,
        "silence_threshold": SILENCE_THRESHOLD
    }
    joblib.dump(profile, "mic_profile.joblib")

def plot_results(all_points):
    plt.subplot(1, 2, 1)
    for sound, points in all_points.items():
        arr = np.array(points)
        plt.scatter(arr[:, 1], arr[:, 0], label=sound)
    plt.xlabel("F2 (Гц)")
    plt.ylabel("F1 (Гц)")
    plt.legend()

    plt.subplot(1, 2, 2)
    for sound, points in all_points.items():
        arr = np.array(points)
        plt.scatter(arr[:, 2], arr[:, 3], label=sound)
    plt.xlabel("Centroid (Гц)")
    plt.ylabel("Ratio")
    plt.legend()

    plt.tight_layout()
    plt.show()

def main():
    all_points={}
    p = None
    try:
        p = pyaudio.PyAudio()
        for i in RECORD_SOUNDS:
            print("Запись звука", i, "на протяжении", DURATION, "сек")
            input("Enter:")
            sound = record_sound(p)
            letter_points=process_recording(sound, i)
            all_points[i] = letter_points
        
        plot_results(all_points)

        X, Y = build_dataset(all_points)
        model, scaler = train_classifier(X, Y)
        #save_profile(model, scaler)       
    finally:
        if p is not None:
            p.terminate()

if __name__=="__main__":
    main()