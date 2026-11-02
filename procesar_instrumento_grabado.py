import numpy as np
import scipy.io.wavfile as wav
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

SAMPLE_RATE = 44100

def aplicar_envolvente(signal, sample_rate, attack_sec=0.08, release_sec=0.08):
    """Aplica fade-in y fade-out para evitar clicks mecánicos al inicio y final."""
    num_samples = len(signal)
    attack = int(sample_rate * attack_sec)
    release = int(sample_rate * release_sec)
    
    envolvente = np.ones(num_samples, dtype=np.float32)
    if attack > 0:
        envolvente[:attack] = np.linspace(0.0, 1.0, attack)
    if release > 0:
        envolvente[-release:] = np.linspace(1.0, 0.0, release)
        
    signal_procesada = signal.astype(np.float32) * envolvente
    return np.clip(signal_procesada, -32768, 32767).astype(np.int16)

def procesar_archivo_grabado(archivo_mp4, prefijo_salida, tiempos_dict, duracion_deseada=1.5):
    """
    Extrae el audio de un vídeo .mp4 y recorta cada muestra según las marcas de tiempo 
    y la duración especificada.
    """
    if not os.path.exists(archivo_mp4):
        print(f"⚠️ No se encontró el archivo '{archivo_mp4}'. Revisa el nombre.")
        return

    archivo_wav_temp = f"temp_extraido_{prefijo_salida}.wav"
    print(f"\n🎬 Extrayendo audio de '{archivo_mp4}'...")
    
    if MOVIEPY_OK:
        audio = AudioFileClip(archivo_mp4)
        audio.write_audiofile(archivo_wav_temp, fps=SAMPLE_RATE, nbytes=2, codec='pcm_s16le', ffmpeg_params=["-ac", "1"])
        audio.close()
    else:
        print("Error: MoviePy no está disponible.")
        return

    sample_rate, data = wav.read(archivo_wav_temp)
    if len(data.shape) > 1:
        data = data.mean(axis=1)

    muestras_por_nota = int(SAMPLE_RATE * duracion_deseada)
    
    print(f"✂️ Recortando muestras con prefijo '{prefijo_salida}' (Duración: {duracion_deseada}s)...")
    for nombre_etiqueta, t_inicio in tiempos_dict.items():
        idx_inicio = int(t_inicio * SAMPLE_RATE)
        idx_fin = idx_inicio + muestras_por_nota
        
        tramo = data[idx_inicio:idx_fin]
        
        # Rellenar con silencio si la toma quedó corta
        if len(tramo) < muestras_por_nota:
            relleno = np.zeros(muestras_por_nota - len(tramo), dtype=data.dtype)
            tramo = np.concatenate((tramo, relleno))
            
        tramo_suave = aplicar_envolvente(tramo, SAMPLE_RATE)
        
        nombre_salida = f"{prefijo_salida}_{nombre_etiqueta}.wav"
        wav.write(nombre_salida, SAMPLE_RATE, tramo_suave)
        print(f"  ✅ Guardada muestra: {nombre_salida} ({duracion_deseada}s)")

    if os.path.exists(archivo_wav_temp):
        os.remove(archivo_wav_temp)

# ==============================================================
# CONFIGURACIÓN DE TUS GRABACIONES Y MARCAS DE TIEMPO
# ==============================================================
if __name__ == '__main__':
    
    # 1. NOTAS INDIVIDUALES (1.0 SEGUNDO)
    procesar_archivo_grabado(
        archivo_mp4="grabacion_piano_notas.mp4",
        prefijo_salida="piano_nota",
        duracion_deseada=1.0,  # 1 SEGUNDO
        tiempos_dict={
                    "DO":  1.2,
                    "RE":  3.6,
                    "MI":  5.5,
                    "FA":  7.6,
                    "SOL": 9.7,
                    "LA":  11.8,
                    "SI":  14.0
                }
    )

    # 2. ACORDES DE PIANO (2.0 SEGUNDOS)
    procesar_archivo_grabado(
        archivo_mp4="grabacion_piano_acordes.mp4",
        prefijo_salida="piano_acorde",
        duracion_deseada=1.0,  # 2 SEGUNDOS
        tiempos_dict={
                    "I_DoM":      2.7,
                    "II_Rem":     5.6,
                    "III_Mim":    9.2,
                    "IV_FaM":     12.2,
                    "V_SolM7":    16.0,
                    "VI_Lam":     19.5,
                    "VII_Sidism": 22.7
                }
    )

    # 3. ACORDES DE ORQUESTA (2.0 SEGUNDOS)
    procesar_archivo_grabado(
        archivo_mp4="grabacion_orquesta_acordes.mp4",
        prefijo_salida="orquesta_acorde",
        duracion_deseada=1.0,  # 2 SEGUNDOS
        tiempos_dict={
                    "I_DoM":      1.2,
                    "II_Rem":     6.0,
                    "III_Mim":    11.5,
                    "IV_FaM":     16.4,
                    "V_SolM7":    20.7,
                    "VI_Lam":     24.8,
                    "VII_Sidism": 29.4
                }
    )

        