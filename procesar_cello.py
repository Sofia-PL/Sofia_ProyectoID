import numpy as np
import scipy.io.wavfile as wav
import math
import os

try:
    from moviepy import AudioFileClip
    MOVIEPY_OK = True
except ImportError:
    try:
        from moviepy.editor import AudioFileClip
        MOVIEPY_OK = True
    except ImportError:
        MOVIEPY_OK = False

# ==============================================================
# CONFIGURACIÓN
# ==============================================================
ARCHIVO_ENTRADA = "grabacion_cello.mp4"  # Coloca aquí el nombre exacto de tu archivo .mp4
DURACION_DESEADA = 1.0                 # Todas las notas durarán exactamente 1.0 segundos
SAMPLE_RATE = 44100                      # Calidad estándar de audio CD

# Tiempos en segundos donde empieza a sonar cada nota en tu grabación .mp4
# Ajusta estos números (en segundos) según el instante en que toques cada nota
TIEMPOS_INICIO_NOTAS = {
    "DO":  2.0,   # Ejemplo: el DO empieza en el segundo 2.0
    "RE":  4.5,   # Ejemplo: el RE empieza en el segundo 5.5
    "MI":  6.5,   # Ejemplo: el MI empieza en el segundo 9.0
    "FA":  8.5,
    "SOL": 10.5,
    "LA":  12.0,
    "SI":  13.9
}

def extraer_audio_wav(archivo_mp4, archivo_wav_salida="audio_completo.wav"):
    """Extrae la pista de audio del archivo .mp4 y la guarda como .wav 44.1kHz Mono."""
    print(f"Extrayendo audio de '{archivo_mp4}'...")
    if MOVIEPY_OK:
        audio = AudioFileClip(archivo_mp4)
        # Convertir a mono 44.1 kHz
        audio.write_audiofile(archivo_wav_salida, fps=SAMPLE_RATE, nbytes=2, codec='pcm_s16le', ffmpeg_params=["-ac", "1"])
        audio.close()
        return archivo_wav_salida
    else:
        print("Error: Instala moviepy usando 'pip install moviepy'")
        return None

def aplicar_envolvente(signal, sample_rate, attack_sec=0.08, release_sec=0.08):
    """Aplica un fade-in y fade-out suave para evitar clicks mecánicos."""
    num_samples = len(signal)
    attack = int(sample_rate * attack_sec)
    release = int(sample_rate * release_sec)
    
    envolvente = np.ones(num_samples, dtype=np.float32)
    
    # Fade in
    if attack > 0:
        envolvente[:attack] = np.linspace(0.0, 1.0, attack)
    # Fade out
    if release > 0:
        envolvente[-release:] = np.linspace(1.0, 0.0, release)
        
    signal_procesada = signal.astype(np.float32) * envolvente
    return np.clip(signal_procesada, -32768, 32767).astype(np.int16)

def recortar_y_homogeneizar_notas(archivo_wav):
    """Corta cada nota a la misma duración exacta y genera las muestras individuales."""
    sample_rate, data = wav.read(archivo_wav)
    
    # Convertir a mono si la grabación es estéreo
    if len(data.shape) > 1:
        data = data.mean(axis=1)

    muestras_por_nota = int(SAMPLE_RATE * DURACION_DESEADA)
    
    print("\nRecortando y ajustando la duración de las notas...")
    for nombre_nota, t_inicio in TIEMPOS_INICIO_NOTAS.items():
        idx_inicio = int(t_inicio * SAMPLE_RATE)
        idx_fin = idx_inicio + muestras_por_nota
        
        # Extraer tramo exacto
        tramo = data[idx_inicio:idx_fin]
        
        # Si la toma queda corta al final, rellenar con silencio
        if len(tramo) < muestras_por_nota:
            relleno = np.zeros(muestras_por_nota - len(tramo), dtype=data.dtype)
            tramo = np.concatenate((tramo, relleno))
            
        # Aplicar suavizado de bordes
        tramo_suave = aplicar_envolvente(tramo, SAMPLE_RATE)
        
        # Guardar archivo individual
        nombre_salida = f"cello_{nombre_nota}.wav"
        wav.write(nombre_salida, SAMPLE_RATE, tramo_suave)
        print(f"  Guardada nota '{nombre_nota}' ({DURACION_DESEADA}s) -> {nombre_salida}")

    print("\n🎉 ¡Todas las notas del violonchelo han sido procesadas con éxito!")

if __name__ == '__main__':
    if os.path.exists(ARCHIVO_ENTRADA):
        wav_extraido = extraer_audio_wav(ARCHIVO_ENTRADA)
        if wav_extraido and os.path.exists(wav_extraido):
            recortar_y_homogeneizar_notas(wav_extraido)
    else:
        print(f" No se encontró el archivo '{ARCHIVO_ENTRADA}'. Asegúrate de colocar tu vídeo .mp4 en la misma carpeta.")