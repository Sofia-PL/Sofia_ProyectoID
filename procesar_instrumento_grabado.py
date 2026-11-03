"""
===============================================================================
 GENERADOR DE MUESTRAS DE AUDIO (violonchelo, piano y orquesta)
===============================================================================
 Lee los vídeos .mp4 de la carpeta  grabaciones/  , extrae su audio, recorta
 cada nota/acorde en el instante indicado y guarda las muestras (.wav) en la
 carpeta  audios/  , que es donde las busca el programa principal.
 
 Estructura (junto a este script):
   grabaciones/  -> grabacion_cello.mp4, grabacion_piano_notas.mp4, ...
   audios/       -> aquí se crean cello_DO.wav, piano_nota_DO.wav, etc.
 
 Requisitos:  pip install moviepy scipy numpy
===============================================================================
"""
 
import os
import tempfile
 
import numpy as np
import scipy.io.wavfile as wav
 
# MoviePy cambió su forma de importar entre versiones; se prueban las dos.
try:
    from moviepy import AudioFileClip
    MOVIEPY_OK = True
except ImportError:
    try:
        from moviepy.editor import AudioFileClip
        MOVIEPY_OK = True
    except ImportError:
        MOVIEPY_OK = False
 
# ***********************************************************************
# CARPETAS Y AJUSTES GENERALES
# ***********************************************************************
DIRECTORIO_SCRIPT = os.path.dirname(os.path.abspath(__file__))
DIR_GRABACIONES = os.path.join(DIRECTORIO_SCRIPT, "grabaciones")   # Entrada: vídeos .mp4
DIR_AUDIOS = os.path.join(DIRECTORIO_SCRIPT, "audios")             # Salida: muestras .wav
 
SAMPLE_RATE = 44100   # Calidad estándar de audio CD
 
# ***********************************************************************
# CONFIGURACIÓN DE GRABACIONES Y MARCAS DE TIEMPO
# ***********************************************************************
# Cada entrada indica:
#   archivo   -> vídeo dentro de la carpeta grabaciones/
#   prefijo   -> inicio del nombre de los .wav de salida (prefijo_ETIQUETA.wav)
#   duracion  -> segundos que durará cada muestra recortada
#   tiempos   -> segundo del vídeo en el que empieza a sonar cada nota/acorde
GRABACIONES = [
    # 1. VIOLONCHELO: notas individuales
    {
        "archivo": "grabacion_cello.mp4",
        "prefijo": "cello",
        "duracion": 1.0,
        "tiempos": {
            "DO":  2.0,
            "RE":  4.5,
            "MI":  6.5,
            "FA":  8.5,
            "SOL": 10.5,
            "LA":  12.0,
            "SI":  13.9,
        },
    },
    # 2. PIANO: notas individuales
    {
        "archivo": "grabacion_piano_notas.mp4",
        "prefijo": "piano_nota",
        "duracion": 1.0,
        "tiempos": {
            "DO":  1.2,
            "RE":  3.6,
            "MI":  5.5,
            "FA":  7.6,
            "SOL": 9.7,
            "LA":  11.8,
            "SI":  14.0,
        },
    },
    # 3. PIANO: acordes
    {
        "archivo": "grabacion_piano_acordes.mp4",
        "prefijo": "piano_acorde",
        "duracion": 1.0,
        "tiempos": {
            "I_DoM":     2.7,
            "II_Rem":    5.6,
            "III_Mim":   9.2,
            "IV_FaM":    12.2,
            "V_SolM":    16.0,
            "VI_Lam":    19.5,
            "VII_Sidim": 22.7,   # Antes se llamaba "VII_Sidism": el programa principal busca "Sidim"
        },
    },
    # 4. ORQUESTA: acordes
    {
        "archivo": "grabacion_orquesta_acordes.mp4",
        "prefijo": "orquesta_acorde",
        "duracion": 1.0,
        "tiempos": {
            "I_DoM":     1.2,
            "II_Rem":    6.0,
            "III_Mim":   11.5,
            "IV_FaM":    16.4,
            "V_SolM":    20.7,
            "VI_Lam":    24.8,
            "VII_Sidim": 29.4,
        },
    },
]
 
 
# ***********************************************************************
# FUNCIONES
# ***********************************************************************
def extraer_audio_wav(archivo_mp4, archivo_wav_salida):
    """Extrae la pista de audio del .mp4 y la guarda como .wav 44.1 kHz mono de 16 bits."""
    audio = AudioFileClip(archivo_mp4)
    audio.write_audiofile(archivo_wav_salida, fps=SAMPLE_RATE, nbytes=2,
                          codec='pcm_s16le', ffmpeg_params=["-ac", "1"])
    audio.close()
 
 
def aplicar_envolvente(signal, sample_rate, attack_sec=0.08, release_sec=0.08):
    """Fade-in y fade-out suaves para evitar clicks al principio y al final de la muestra."""
    attack = int(sample_rate * attack_sec)
    release = int(sample_rate * release_sec)
 
    envolvente = np.ones(len(signal), dtype=np.float32)
    if attack > 0:
        envolvente[:attack] = np.linspace(0.0, 1.0, attack)      # Fade in
    if release > 0:
        envolvente[-release:] = np.linspace(1.0, 0.0, release)   # Fade out
 
    procesada = signal.astype(np.float32) * envolvente
    return np.clip(procesada, -32768, 32767).astype(np.int16)
 
 
def recortar_muestras(archivo_wav, prefijo, tiempos, duracion):
    """Corta cada nota a la misma duración exacta y guarda un .wav por nota en /audios."""
    _, data = wav.read(archivo_wav)
 
    # Pasa a mono si la grabación fuera estéreo
    if len(data.shape) > 1:
        data = data.mean(axis=1)
 
    muestras_por_nota = int(SAMPLE_RATE * duracion)
 
    for etiqueta, t_inicio in tiempos.items():
        idx_inicio = int(t_inicio * SAMPLE_RATE)
        tramo = data[idx_inicio:idx_inicio + muestras_por_nota]
 
        if len(tramo) == 0:
            print(f"   AVISO: el segundo {t_inicio} de '{etiqueta}' está fuera de la grabación (se guarda en silencio).")
 
        # Si la toma queda corta al final, se rellena con silencio
        if len(tramo) < muestras_por_nota:
            relleno = np.zeros(muestras_por_nota - len(tramo), dtype=data.dtype)
            tramo = np.concatenate((tramo, relleno))
 
        nombre_salida = f"{prefijo}_{etiqueta}.wav"
        wav.write(os.path.join(DIR_AUDIOS, nombre_salida), SAMPLE_RATE, aplicar_envolvente(tramo, SAMPLE_RATE))
        print(f"   Guardada: audios/{nombre_salida} ({duracion}s)")
 
 
def procesar_grabacion(config):
    """Procesa una grabación completa: extrae el audio, lo recorta y borra el temporal."""
    ruta_mp4 = os.path.join(DIR_GRABACIONES, config["archivo"])
    if not os.path.exists(ruta_mp4):
        print(f"\nNo se encontró 'grabaciones/{config['archivo']}'. Se omite.")
        return
 
    print(f"\nExtrayendo audio de 'grabaciones/{config['archivo']}'...")
    ruta_temp = os.path.join(tempfile.gettempdir(), f"temp_extraido_{config['prefijo']}.wav")
    try:
        extraer_audio_wav(ruta_mp4, ruta_temp)
        print(f"Recortando muestras '{config['prefijo']}' (duración: {config['duracion']}s)...")
        recortar_muestras(ruta_temp, config["prefijo"], config["tiempos"], config["duracion"])
    finally:
        if os.path.exists(ruta_temp):
            os.remove(ruta_temp)   # El audio completo extraído no se necesita después
 
 
def main():
    if not MOVIEPY_OK:
        print("Error: instala moviepy con 'pip install moviepy'")
        return
 
    os.makedirs(DIR_GRABACIONES, exist_ok=True)
    os.makedirs(DIR_AUDIOS, exist_ok=True)
 
    for config in GRABACIONES:
        procesar_grabacion(config)
 
    print("\nProceso terminado. Las muestras están en la carpeta 'audios'.")
 
 
if __name__ == '__main__':
    main()
 