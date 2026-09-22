import numpy as np
import librosa
from scipy.signal import find_peaks
import joblib

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

def extract_formants(block, RATE):
    """
    Функция извлечения формант F1 и F2 из звукового блока
    На вход подается звуковой блок и частота дискретизации, на выходе F1, F2, centroid, ratio и RMS этого блока
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

def load_profile(filename):
    profile = joblib.load(filename)
    return profile["model"], profile["scaler"], profile["silence_threshold"]

def predict_sound(block, model, scaler, silence_threshold, RATE):
    """
    На входе: блок аудио 50мс, модель, scaler, порог тишины, частота дискретизации.
    На выходе: строка с распознанным звуком, либо "-" при тишине.
    """
    f1, f2, centroid, ratio, rms = extract_formants(block, RATE)
    if rms <= silence_threshold:
        return "-"
    sound = model.predict(scaler.transform([[f1, f2, centroid, ratio]]))
    return sound[0]