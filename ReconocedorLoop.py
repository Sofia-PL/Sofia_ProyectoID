"""
===============================================================================
 SINTETIZADOR GESTUAL OPENCV + LOOPSTATION SINCRONIZADA
===============================================================================
 Funcionamiento general:
   - La cámara detecta 4 fichas de colores (rosa, azul, verde y roja).
   - Según qué fichas estén TAPADAS o VISIBLES se decide una nota o un acorde.
   - El sonido se reproduce con pygame (o winsound como alternativa en Windows).
   - Hay un metrónomo a 60 BPM y una loopstation que graba sincronizada a él.

 Controles:
   T -> activa/desactiva el metrónomo
   I -> cambia de instrumento
   M -> alterna entre modo ACORDES y modo NOTAS INDIVIDUALES
   R -> loopstation: armar / cerrar y cuantizar grabación
   C -> borra el loop (Clear)
   P -> empieza/termina la grabación de partitura y exporta 3 láminas
   H -> abre el tutorial
   Q / ESC -> salir

 Láminas del tutorial: lamina_1 ... lamina_5 (.png/.jpg) en la misma carpeta
 que este script. Si no existen, se dibuja un tutorial de texto de reserva.
===============================================================================
"""

import cv2          # Visión artificial: cámara, máscaras de color, dibujo y ventanas
import numpy as np  # Arrays para las imágenes y los rangos de color
import struct       # Para construir la cabecera de los archivos WAV a mano
import math         # Funciones seno, exponenciales y redondeos
import time         # Cronómetros (metrónomo, estabilidad de notas, loop)
import os           # Comprobar si existen archivos

# ------------------------------------------------------------------
# Carga de librerías de audio (ambas son opcionales)
# ------------------------------------------------------------------
# pygame permite reproducir varios sonidos a la vez en canales distintos.
try:
    import pygame
    PYGAME_DISPONIBLE = True
except ImportError:
    PYGAME_DISPONIBLE = False

# winsound sólo existe en Windows; sirve como alternativa simple.
try:
    import winsound
    WINSOUND_DISPONIBLE = True
except ImportError:
    WINSOUND_DISPONIBLE = False

# Carpeta donde está este script (aquí se buscan las láminas e imágenes)
DIRECTORIO_SCRIPT = os.path.dirname(os.path.abspath(__file__))

# ==========================================
# 1. DEFINICIÓN DE FRECUENCIAS Y MAPEOS (ESTADO FUNDAMENTAL)
# ==========================================

# Notas individuales (octava central). Cada una es una lista con una frecuencia en Hz.
NOTAS_SOLO = {
    "DO":  [261.63],
    "RE":  [293.66],
    "MI":  [329.63],
    "FA":  [349.23],
    "SOL": [392.00],
    "LA":  [440.00],
    "SI":  [493.88]
}

# Acordes en estado fundamental (tríadas directas sobre cada grado de la escala de Do)
ACORDES = {
    "I (Do Mayor)":     [261.63, 329.63, 392.00],   # C4 - E4 - G4
    "II (Re menor)":    [293.66, 349.23, 440.00],   # D4 - F4 - A4
    "III (Mi menor)":   [329.63, 392.00, 493.88],   # E4 - G4 - B4
    "IV (Fa Mayor)":    [349.23, 440.00, 523.25],   # F4 - A4 - C5
    "V (Sol Mayor)":    [392.00, 493.88, 587.33],   # G4 - B4 - D5
    "VI (La menor)":    [440.00, 523.25, 659.25],   # A4 - C5 - E5
    "VII (Si dim)":     [493.88, 587.33, 698.46]    # B4 - D5 - F5
}

# Nombre del acorde -> etiqueta para nombres de archivo de audio/imagen
MAPEO_CLAVE_ACORDE_ARCHIVO = {
    "I (Do Mayor)":     "I_DoM",
    "II (Re menor)":    "II_Rem",
    "III (Mi menor)":   "III_Mim",
    "IV (Fa Mayor)":    "IV_FaM",
    "V (Sol Mayor)":    "V_SolM",
    "VI (La menor)":    "VI_Lam",
    "VII (Si dim)":     "VII_Sidim"
}

# Nombre del acorde -> etiqueta corta que se muestra en pantalla
ETIQUETA_CORTA_ACORDE = {
    "I (Do Mayor)":     "DoM",
    "II (Re menor)":    "Rem",
    "III (Mi menor)":   "Mim",
    "IV (Fa Mayor)":    "FaM",
    "V (Sol Mayor)":    "SolM",
    "VI (La menor)":    "Lam",
    "VII (Si dim)":     "Sidim"
}

# Nota individual -> etiqueta de acorde (para buscar la imagen de piano de esa nota)
NOTA_A_TAG_ACORDE = {
    "DO":  "I_DoM",
    "RE":  "II_Rem",
    "MI":  "III_Mim",
    "FA":  "IV_FaM",
    "SOL": "V_SolM",
    "LA":  "VI_Lam",
    "SI":  "VII_Sidim"
}

# Nombre del acorde -> nota sobre la que está construido (su fundamental)
ACORDE_A_NOTA = {
    "I (Do Mayor)":     "DO",
    "II (Re menor)":    "RE",
    "III (Mi menor)":   "MI",
    "IV (Fa Mayor)":    "FA",
    "V (Sol Mayor)":    "SOL",
    "VI (La menor)":    "LA",
    "VII (Si dim)":     "SI"
}

# Qué fichas están VISIBLES para cada nota: (rosa, azul, verde, roja)
# Sirve para dibujar la tablatura de fichas de una nota ya tocada.
VISIBILIDAD_POR_NOTA = {
    "DO":  (True,  True,  True,  True),
    "RE":  (False, True,  True,  True),
    "MI":  (True,  False, True,  True),
    "FA":  (True,  True,  False, True),
    "SOL": (True,  True,  True,  False),
    "LA":  (False, False, True,  True),
    "SI":  (False, True,  False, True)
}

# Rutas de los archivos .wav de cada instrumento (se rellenan al arrancar)
RUTAS_AUDIO = {
    "SINTETIZADOR": {"NOTAS": {}, "ACORDES": {}},
    "PIANO":        {"NOTAS": {}, "ACORDES": {}},
    "VIOLONCHELO":  {"NOTAS": {}, "ACORDES": {}},
    "ORQUESTA":     {"NOTAS": {}, "ACORDES": {}}
}

# Objetos pygame.mixer.Sound ya cargados en memoria (misma estructura)
SONIDOS_PYGAME = {
    "SINTETIZADOR": {"NOTAS": {}, "ACORDES": {}},
    "PIANO":        {"NOTAS": {}, "ACORDES": {}},
    "VIOLONCHELO":  {"NOTAS": {}, "ACORDES": {}},
    "ORQUESTA":     {"NOTAS": {}, "ACORDES": {}}
}

SONIDO_METRONOMO = None   # Sonido del "click" del metrónomo
canal_loop = None         # Canal de pygame para el loop grabado
canal_vivo = None         # Canal de pygame para lo que tocas en directo

# Valor del trackbar de brillo (100 = sin cambio, <100 oscurece, >100 aclara)
incremento_brillo = 100


def actualizar_brillo_trackbar(val):
    """Callback del trackbar: guarda el valor actual del brillo."""
    global incremento_brillo
    incremento_brillo = val


def ajustar_brillo_frame(img, ajuste):
    """Aumenta o reduce el brillo del frame modificando el canal V (HSV).
    Se evita desbordar el rango 0-255 de uint8 saturando los valores."""
    offset = ajuste - 100
    if offset == 0:
        return img
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)
    if offset > 0:
        # Los píxeles que superarían 255 se fijan en 255; el resto suma el offset
        limite = 255 - offset
        v[v > limite] = 255
        v[v <= limite] += offset
    else:
        # Los píxeles que bajarían de 0 se fijan en 0; el resto resta el offset
        abs_offset = abs(offset)
        v[v < abs_offset] = 0
        v[v >= abs_offset] -= abs_offset
    hsv = cv2.merge((h, s, v))
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)


def mostrar_imagen_segura(titulo, img):
    """imshow que no rompe el programa si la ventana no se puede abrir."""
    try:
        cv2.imshow(titulo, img)
    except Exception:
        pass


# ==========================================
# 2. SÍNTESIS Y CARGA DE AUDIOS
# ==========================================
def generar_bytes_wav(lista_frecuencias, duracion=1.0, sample_rate=44100, estilo="SYNTH"):
    """Genera en memoria un archivo WAV (mono, 16 bits) con una o varias
    frecuencias sonando a la vez.

    estilo: "METRONOMO", "PIANO", "ORQUESTA" o cualquier otro (timbre de sintetizador).
    Devuelve los bytes completos: cabecera WAV + muestras.
    """
    num_samples = int(sample_rate * duracion)
    data_size = num_samples * 2  # 2 bytes por muestra (16 bits)

    # Cabecera estándar de un WAV PCM mono de 16 bits
    header = struct.pack(
        '<4sI4s4sIHHIIHH4sI',
        b'RIFF', 36 + data_size, b'WAVE', b'fmt ',
        16, 1, 1, sample_rate, sample_rate * 2, 2, 16,
        b'data', data_size
    )
    samples = bytearray()

    # Aceptamos tanto una lista de frecuencias como una sola
    if isinstance(lista_frecuencias, (list, tuple)):
        freqs = list(lista_frecuencias)
    else:
        freqs = [lista_frecuencias]

    num_notas = len(freqs)

    if estilo == "METRONOMO":
        # Click corto: un seno que se apaga linealmente
        freq = freqs[0] if len(freqs) > 0 else 1000.0
        for i in range(num_samples):
            t = i / sample_rate
            env = 1.0 - (i / num_samples)
            val = int(math.sin(2 * math.pi * freq * t) * env * 24000)
            val = max(-32768, min(32767, val))
            samples.extend(struct.pack('<h', val))

    elif estilo == "PIANO":
        # Ataque muy rápido y caída exponencial, con 4 armónicos
        attack = int(sample_rate * 0.005)
        for i in range(num_samples):
            t = i / sample_rate
            envolvente = math.exp(-3.5 * t)
            if i < attack:
                envolvente *= (i / attack)
            mezcla_onda = 0.0
            for freq in freqs:
                mezcla_onda += (1.00 * math.sin(2 * math.pi * freq * t) +
                                0.45 * math.sin(2 * math.pi * 2 * freq * t) +
                                0.20 * math.sin(2 * math.pi * 3 * freq * t) +
                                0.10 * math.sin(2 * math.pi * 4 * freq * t))
            mezcla_onda /= num_notas  # Normaliza para que un acorde no sature
            val = int(mezcla_onda * envolvente * 16000)
            val = max(-32768, min(32767, val))
            samples.extend(struct.pack('<h', val))

    elif estilo == "ORQUESTA":
        # Ataque y relajación lentos + vibrato + voces ligeramente desafinadas
        # (efecto "coro de cuerdas")
        attack = int(sample_rate * 0.12)
        release = int(sample_rate * 0.12)
        for i in range(num_samples):
            t = i / sample_rate
            envolvente = 0.80
            if i < attack:
                envolvente = (i / attack) * 0.80
            elif i > num_samples - release:
                envolvente = ((num_samples - i) / release) * 0.80
            mezcla_onda = 0.0
            vibrato = 1.0 + 0.004 * math.sin(2 * math.pi * 5.0 * t)
            for freq in freqs:
                f_mod = freq * vibrato
                mezcla_onda += (0.50 * math.sin(2 * math.pi * f_mod * t) +
                                0.25 * math.sin(2 * math.pi * (f_mod * 1.003) * t) +
                                0.25 * math.sin(2 * math.pi * (f_mod * 0.997) * t) +
                                0.35 * math.sin(2 * math.pi * 2 * f_mod * t) +
                                0.15 * math.sin(2 * math.pi * 0.5 * f_mod * t))
            mezcla_onda /= num_notas
            val = int(mezcla_onda * envolvente * 16000)
            val = max(-32768, min(32767, val))
            samples.extend(struct.pack('<h', val))

    else:
        # Sintetizador básico: fundamental + segundo armónico y envolvente suave
        attack = int(sample_rate * 0.08)
        release = int(sample_rate * 0.08)
        for i in range(num_samples):
            t = i / sample_rate
            mezcla_onda = 0.0
            for freq in freqs:
                mezcla_onda += (0.65 * math.sin(2 * math.pi * freq * t) +
                                0.35 * math.sin(2 * math.pi * 2 * freq * t))
            mezcla_onda /= num_notas
            envolvente = 0.85
            if i < attack:
                envolvente = (i / attack) * 0.85
            elif i > num_samples - release:
                envolvente = ((num_samples - i) / release) * 0.85
            val = int(mezcla_onda * envolvente * 16000)
            val = max(-32768, min(32767, val))
            samples.extend(struct.pack('<h', val))

    return header + samples


def registrar_audio(inst, tipo, clave, ruta, freqs_fallback=None):
    """Registra el audio de un instrumento para una nota/acorde.

    1) Si existe el archivo real (ej. piano_nota_DO.wav) lo usa.
    2) Si no existe y hay frecuencias de reserva, genera un WAV sintético.
    3) Si no hay frecuencias, reutiliza el sonido del sintetizador.
    """
    if os.path.exists(ruta):
        RUTAS_AUDIO[inst][tipo][clave] = ruta
        if PYGAME_DISPONIBLE:
            try:
                SONIDOS_PYGAME[inst][tipo][clave] = pygame.mixer.Sound(ruta)
            except Exception:
                pass
    else:
        if freqs_fallback:
            data = generar_bytes_wav(freqs_fallback, duracion=1.0, estilo=inst)
            ruta_gen = f"temp_{inst.lower()}_{tipo.lower()}_{clave}.wav".replace(" ", "_").replace("(", "").replace(")", "")
            with open(ruta_gen, "wb") as f:
                f.write(data)
            RUTAS_AUDIO[inst][tipo][clave] = ruta_gen
            if PYGAME_DISPONIBLE:
                try:
                    SONIDOS_PYGAME[inst][tipo][clave] = pygame.mixer.Sound(ruta_gen)
                except Exception:
                    pass
        else:
            ruta_synth = RUTAS_AUDIO["SINTETIZADOR"][tipo].get(clave)
            if ruta_synth and os.path.exists(ruta_synth):
                RUTAS_AUDIO[inst][tipo][clave] = ruta_synth
                if PYGAME_DISPONIBLE:
                    SONIDOS_PYGAME[inst][tipo][clave] = SONIDOS_PYGAME["SINTETIZADOR"][tipo].get(clave)


def pregenerar_sonidos():
    """Inicializa el mezclador de audio y deja listos todos los sonidos
    (metrónomo, sintetizador, violonchelo, piano y orquesta)."""
    global PYGAME_DISPONIBLE, canal_loop, canal_vivo, SONIDO_METRONOMO

    if PYGAME_DISPONIBLE:
        try:
            # Buffer pequeño (512) = poca latencia al tocar
            pygame.mixer.pre_init(44100, -16, 1, 512)
            pygame.mixer.init()
            pygame.mixer.set_num_channels(8)
            canal_loop = pygame.mixer.Channel(0)   # Canal 0: loop grabado
            canal_vivo = pygame.mixer.Channel(1)   # Canal 1: nota en directo
            print("Audio Pygame multicanal inicializado correctamente.")
        except Exception as e:
            print(f"Uso de audio estándar: {e}")
            PYGAME_DISPONIBLE = False

    # Click del metrónomo (50 ms a 1000 Hz)
    data_metro = generar_bytes_wav([1000.0], duracion=0.05, estilo="METRONOMO")
    ruta_metro = "temp_metronomo_click.wav"
    with open(ruta_metro, "wb") as f:
        f.write(data_metro)
    if PYGAME_DISPONIBLE:
        try:
            SONIDO_METRONOMO = pygame.mixer.Sound(ruta_metro)
        except Exception:
            SONIDO_METRONOMO = None

    # Sintetizador: notas individuales
    for nombre, freqs in NOTAS_SOLO.items():
        data = generar_bytes_wav(freqs, duracion=1.0, estilo="SYNTH")
        ruta = f"temp_synth_nota_{nombre}.wav"
        with open(ruta, "wb") as f:
            f.write(data)
        RUTAS_AUDIO["SINTETIZADOR"]["NOTAS"][nombre] = ruta
        if PYGAME_DISPONIBLE:
            SONIDOS_PYGAME["SINTETIZADOR"]["NOTAS"][nombre] = pygame.mixer.Sound(ruta)

    # Sintetizador: acordes
    for nombre, freqs in ACORDES.items():
        clave_limpia = "".join(e for e in nombre if e.isalnum())
        data = generar_bytes_wav(freqs, duracion=1.0, estilo="SYNTH")
        ruta = f"temp_synth_acorde_{clave_limpia}.wav"
        with open(ruta, "wb") as f:
            f.write(data)
        RUTAS_AUDIO["SINTETIZADOR"]["ACORDES"][nombre] = ruta
        if PYGAME_DISPONIBLE:
            SONIDOS_PYGAME["SINTETIZADOR"]["ACORDES"][nombre] = pygame.mixer.Sound(ruta)

    # Violonchelo (sólo en modo notas): usa cello_DO.wav ... si existen
    for nombre, freqs in NOTAS_SOLO.items():
        registrar_audio("VIOLONCHELO", "NOTAS", nombre, f"cello_{nombre}.wav", freqs_fallback=freqs)

    # Piano (notas): piano_nota_DO.wav ... si existen
    for nombre, freqs in NOTAS_SOLO.items():
        registrar_audio("PIANO", "NOTAS", nombre, f"piano_nota_{nombre}.wav", freqs_fallback=freqs)

    # Piano (acordes): piano_acorde_I_DoM.wav ... si existen
    for nombre, freqs in ACORDES.items():
        tag = MAPEO_CLAVE_ACORDE_ARCHIVO[nombre]
        registrar_audio("PIANO", "ACORDES", nombre, f"piano_acorde_{tag}.wav", freqs_fallback=freqs)

    # Orquesta (sólo en modo acordes): orquesta_acorde_I_DoM.wav ... si existen
    for nombre, freqs in ACORDES.items():
        tag = MAPEO_CLAVE_ACORDE_ARCHIVO[nombre]
        registrar_audio("ORQUESTA", "ACORDES", nombre, f"orquesta_acorde_{tag}.wav", freqs_fallback=freqs)

    print("Sintetizador y muestras de instrumentos inicializados correctamente.")


# ==========================================
# 3. REPRODUCCIÓN MULTICANAL
# ==========================================
def sonar_click_metronomo():
    """Reproduce el click del metrónomo (pygame o, si no, un beep de winsound)."""
    if PYGAME_DISPONIBLE and SONIDO_METRONOMO:
        try:
            SONIDO_METRONOMO.play()
        except Exception:
            pass
    elif WINSOUND_DISPONIBLE:
        try:
            winsound.Beep(1000, 40)
        except Exception:
            pass


def obtener_recurso_audio(clave_actual, modo_acordes, instrumento):
    """Devuelve el sonido (objeto pygame) o la ruta (winsound) correspondiente a
    la nota/acorde y al instrumento. Si el instrumento no lo tiene, usa el
    sintetizador como reserva."""
    if clave_actual == "SILENCIO":
        return None
    tipo = "ACORDES" if modo_acordes else "NOTAS"

    if PYGAME_DISPONIBLE:
        snd = SONIDOS_PYGAME.get(instrumento, {}).get(tipo, {}).get(clave_actual)
        if snd:
            return snd
        return SONIDOS_PYGAME["SINTETIZADOR"][tipo].get(clave_actual)
    else:
        ruta = RUTAS_AUDIO.get(instrumento, {}).get(tipo, {}).get(clave_actual)
        if ruta:
            return ruta
        return RUTAS_AUDIO["SINTETIZADOR"][tipo].get(clave_actual)


def reproducir_sonido_vivo(clave_actual, modo_acordes, instrumento):
    """Reproduce en bucle la nota/acorde que estás tocando ahora mismo.
    Si la clave es SILENCIO, corta el sonido."""
    if clave_actual == "SILENCIO":
        detener_audio_vivo()
        return

    recurso = obtener_recurso_audio(clave_actual, modo_acordes, instrumento)
    if PYGAME_DISPONIBLE and canal_vivo and recurso:
        canal_vivo.play(recurso, loops=-1)  # loops=-1: se repite hasta pararlo
    elif WINSOUND_DISPONIBLE and recurso and isinstance(recurso, str):
        try:
            winsound.PlaySound(recurso, winsound.SND_FILENAME | winsound.SND_ASYNC | winsound.SND_LOOP)
        except Exception:
            pass


def reproducir_sonido_loop(clave, modo_acordes, instrumento):
    """Igual que el anterior pero en el canal del loop grabado."""
    if clave == "SILENCIO":
        detener_audio_loop()
        return

    recurso = obtener_recurso_audio(clave, modo_acordes, instrumento)
    if PYGAME_DISPONIBLE and canal_loop and recurso:
        canal_loop.play(recurso, loops=-1)
    elif WINSOUND_DISPONIBLE and recurso and isinstance(recurso, str):
        try:
            winsound.PlaySound(recurso, winsound.SND_FILENAME | winsound.SND_ASYNC | winsound.SND_LOOP)
        except Exception:
            pass


def detener_audio_vivo():
    """Para el sonido en directo."""
    if PYGAME_DISPONIBLE and canal_vivo:
        canal_vivo.stop()
    elif WINSOUND_DISPONIBLE:
        try:
            winsound.PlaySound(None, winsound.SND_PURGE)
        except Exception:
            pass


def detener_audio_loop():
    """Para el sonido del loop (con winsound no hay canales separados)."""
    if PYGAME_DISPONIBLE and canal_loop:
        canal_loop.stop()


# ==========================================
# 4. RANGOS DE COLOR HSV Y VISIBILIDAD
# ==========================================
# En OpenCV el matiz (H) va de 0 a 180. Cada ficha se detecta con un rango
# [bajo, alto] de (H, S, V). S y V mínimos de 50 descartan grises y zonas oscuras.
rosa_bajo = np.array([138, 50, 50])
rosa_alto = np.array([168, 255, 255])

azul_bajo = np.array([95, 50, 50])
azul_alto = np.array([130, 255, 255])

verde_bajo = np.array([35, 50, 50])
verde_alto = np.array([85, 255, 255])

# El rojo está en los dos extremos del círculo de matices, así que usa dos rangos
rojo_bajo1 = np.array([0, 50, 50])
rojo_alto1 = np.array([8, 255, 255])
rojo_bajo2 = np.array([172, 50, 50])
rojo_alto2 = np.array([180, 255, 255])

# Área mínima (en píxeles) para considerar que una ficha está visible
UMBRAL_AREA_MINIMA = 3000


def evaluar_visibilidad(hsv_frame, bajo, alto, bajo2=None, alto2=None):
    """Decide si una ficha de color es visible en el frame.

    Devuelve (visible, area_maxima, contorno_mayor).
    """
    # Máscara binaria: blanco donde el píxel está dentro del rango de color
    mascara = cv2.inRange(hsv_frame, bajo, alto)
    if bajo2 is not None and alto2 is not None:
        # Segundo rango (para el rojo) y unión de ambas máscaras
        mascara2 = cv2.inRange(hsv_frame, bajo2, alto2)
        mascara = cv2.bitwise_or(mascara, mascara2)

    # Apertura morfológica: elimina ruido pequeño y recupera el tamaño
    mascara = cv2.erode(mascara, None, iterations=2)
    mascara = cv2.dilate(mascara, None, iterations=2)

    contornos, _ = cv2.findContours(mascara, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

    area_maxima = 0
    contorno_max = None
    if len(contornos) > 0:
        contorno_max = max(contornos, key=cv2.contourArea)
        area_maxima = cv2.contourArea(contorno_max)

    visible = area_maxima >= UMBRAL_AREA_MINIMA
    return visible, area_maxima, contorno_max


# ==========================================
# 5. BÚSQUEDA DE IMÁGENES Y DIBUJO DE TARJETAS
# ==========================================
def leer_imagen_segura(ruta):
    """Lee una imagen del disco. Usa imdecode para que funcione también con
    rutas con tildes o caracteres especiales en Windows. Devuelve None si falla."""
    try:
        datos = np.fromfile(ruta, dtype=np.uint8)
        return cv2.imdecode(datos, cv2.IMREAD_COLOR)
    except Exception:
        return None


def cargar_imagen_lamina(nombre_base):
    """Busca una lámina del tutorial (ej. 'lamina_1') con extensión .png, .jpg,
    .jpeg, .webp o .bmp, en la carpeta del script y en la carpeta actual.
    Devuelve la imagen o None si no la encuentra.

    (Esta función faltaba en el código original y causaba el NameError.)
    """
    carpetas = []
    for c in (DIRECTORIO_SCRIPT, os.getcwd()):
        if c not in carpetas:
            carpetas.append(c)

    # Variantes del nombre por si el archivo está en mayúsculas
    nombres = []
    for n in (nombre_base, nombre_base.lower(), nombre_base.capitalize(), nombre_base.upper()):
        if n not in nombres:
            nombres.append(n)

    extensiones = [".png", ".jpg", ".jpeg", ".webp", ".bmp",
                   ".PNG", ".JPG", ".JPEG", ".WEBP", ".BMP"]

    for carpeta in carpetas:
        for nombre in nombres:
            for ext in extensiones:
                ruta = os.path.join(carpeta, nombre + ext)
                if os.path.isfile(ruta):
                    img = leer_imagen_segura(ruta)
                    if img is not None:
                        return img
    return None


def buscar_imagen_tablatura(tag_nombre):
    """Busca la imagen de piano de un acorde/nota probando muchos nombres de
    archivo posibles (tab_I_DoM.png, tab_DoM.jpg, sidim.png, etc.)."""
    extensiones = [".png", ".jpg", ".jpeg", ".PNG", ".JPG", ".JPEG"]

    posibles_bases = [
        f"tab_{tag_nombre}",
        f"tab_{tag_nombre.lower()}",
        f"tab_{tag_nombre.upper()}",
        f"{tag_nombre}",
        f"{tag_nombre.lower()}"
    ]

    # Si el tag es del tipo "I_DoM", probamos también sólo con "DoM"
    if "_" in tag_nombre:
        subtag = tag_nombre.split("_")[-1]
        posibles_bases.extend([
            f"tab_{subtag}",
            f"tab_{subtag.lower()}",
            f"tab_{subtag.upper()}",
            f"tab_VII_{subtag}",
            f"tab_VII_{subtag.lower()}"
        ])
        # Caso especial del acorde disminuido (varios nombres posibles)
        if "sidim" in subtag.lower() or "sidism" in subtag.lower():
            posibles_bases.extend([
                "tab_Sidim", "tab_sidim", "Sidim", "sidim",
                "tab_VII_Sidim", "tab_VII_sidim"
            ])

    for base in posibles_bases:
        for ext in extensiones:
            ruta = os.path.join(DIRECTORIO_SCRIPT, f"{base}{ext}")
            if not os.path.exists(ruta):
                ruta = f"{base}{ext}"  # También en la carpeta de trabajo actual
            if os.path.exists(ruta):
                img = leer_imagen_segura(ruta)
                if img is not None:
                    return img
    return None


def dibujar_piano_esquematico(ancho=280, alto=130):
    """Dibuja un teclado de 7 teclas blancas y 5 negras (reserva cuando no hay
    imagen de piano para esa nota/acorde)."""
    img = np.full((alto, ancho, 3), 255, dtype=np.uint8)
    cv2.rectangle(img, (0, 0), (ancho - 1, alto - 1), (0, 0, 0), 2)

    # Teclas blancas
    num_teclas_blancas = 7
    w_tb = ancho // num_teclas_blancas
    for i in range(num_teclas_blancas):
        x = i * w_tb
        cv2.rectangle(img, (x, 0), (x + w_tb, alto - 1), (0, 0, 0), 1)

    # Teclas negras: aparecen tras Do, Re, Fa, Sol y La (no tras Mi ni Si)
    h_tn = int(alto * 0.6)
    w_tn = int(w_tb * 0.6)
    indices_negras = [0, 1, 3, 4, 5]
    for idx in indices_negras:
        x_n = (idx + 1) * w_tb - (w_tn // 2)
        cv2.rectangle(img, (x_n, 0), (x_n + w_tn, h_tn), (0, 0, 0), -1)

    return img


def obtener_visibilidad_de_clave(clave):
    """Devuelve (rosa, azul, verde, roja) visibles para una nota o un acorde.
    Así la tablatura refleja la nota ya tocada y no el estado actual de la cámara."""
    nota = ACORDE_A_NOTA.get(clave, clave)
    return VISIBILIDAD_POR_NOTA.get(nota, (True, True, True, True))


def generar_tarjeta_fichas(clave, modo_acordes):
    """Tarjeta 220x160 con las 4 fichas (visible = círculo de color,
    tapada = círculo gris con X roja) y el nombre de la nota/acorde."""
    rosa_vis, azul_vis, verde_vis, roja_vis = obtener_visibilidad_de_clave(clave)

    tarjeta = np.zeros((160, 220, 3), dtype=np.uint8)
    cv2.rectangle(tarjeta, (0, 0), (220, 160), (255, 255, 255), 2)

    # (texto, visible, centro, color BGR)
    fichas_coords = [
        ("RS", rosa_vis, (30, 40), (255, 0, 255)),
        ("A", azul_vis, (80, 40), (255, 0, 0)),
        ("V", verde_vis, (130, 40), (0, 255, 0)),
        ("RJ", roja_vis, (180, 40), (0, 0, 255))
    ]

    for nombre_f, visible, (cx, cy), color_bgr in fichas_coords:
        if visible:
            cv2.circle(tarjeta, (cx, cy), 16, color_bgr, -1)
            cv2.putText(tarjeta, nombre_f, (cx - 7, cy + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (255, 255, 255), 1)
        else:
            cv2.circle(tarjeta, (cx, cy), 16, (60, 60, 60), -1)
            cv2.line(tarjeta, (cx - 10, cy - 10), (cx + 10, cy + 10), (0, 0, 255), 2)
            cv2.line(tarjeta, (cx - 10, cy + 10), (cx + 10, cy - 10), (0, 0, 255), 2)

    # Recuadro inferior con el nombre
    cv2.rectangle(tarjeta, (10, 85), (210, 150), (30, 30, 30), -1)
    cv2.rectangle(tarjeta, (10, 85), (210, 150), (180, 180, 180), 1)

    lbl_corta = ETIQUETA_CORTA_ACORDE.get(clave, clave)
    texto = lbl_corta if modo_acordes else f"NOTA: {clave}"
    cv2.putText(tarjeta, texto, (15, 122), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)
    return tarjeta


def generar_tarjeta_piano(clave, modo_acordes):
    """Tarjeta 300x210 con el dibujo de piano de la nota/acorde (imagen real si
    existe, o un teclado esquemático) y su nombre debajo."""
    tarjeta = np.full((210, 300, 3), 255, dtype=np.uint8)
    cv2.rectangle(tarjeta, (0, 0), (300, 210), (0, 0, 0), 2)

    if modo_acordes:
        tag_busqueda = MAPEO_CLAVE_ACORDE_ARCHIVO.get(clave, "I_DoM")
        texto_etiqueta = ETIQUETA_CORTA_ACORDE.get(clave, clave)
    else:
        tag_busqueda = NOTA_A_TAG_ACORDE.get(clave, "I_DoM")
        texto_etiqueta = f"NOTA: {clave}"

    img_piano = buscar_imagen_tablatura(tag_busqueda)

    if img_piano is not None:
        # Escala la imagen para que quepa en 280x130 sin deformarse
        h, w = img_piano.shape[:2]
        target_w, target_h = 280, 130
        scale = min(target_w / w, target_h / h)
        nw, nh = int(w * scale), int(h * scale)
        img_resized = cv2.resize(img_piano, (nw, nh))

        # La centra dentro del recuadro
        off_x = 10 + (target_w - nw) // 2
        off_y = 15 + (target_h - nh) // 2
        tarjeta[off_y:off_y + nh, off_x:off_x + nw] = img_resized
        cv2.rectangle(tarjeta, (10, 15), (290, 150), (180, 180, 180), 1)
    else:
        piano_synth = dibujar_piano_esquematico(280, 130)
        tarjeta[15:145, 10:290] = piano_synth
        cv2.rectangle(tarjeta, (10, 15), (290, 150), (0, 0, 0), 1)

    cv2.rectangle(tarjeta, (10, 158), (290, 198), (30, 30, 30), -1)
    cv2.putText(tarjeta, texto_etiqueta, (15, 183), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 1)

    return tarjeta


def registrar_nota_partitura(clave, modo_acordes, duracion, eventos, cards_fichas, cards_piano):
    """Guarda una nota tocada en los tres resultados: el evento rítmico para el
    pentagrama y las tarjetas de las dos tablaturas. Mantener los tres juntos
    evita que las láminas queden con distinto número de notas."""
    eventos.append((clave, modo_acordes, duracion))
    cards_fichas.append(generar_tarjeta_fichas(clave, modo_acordes))
    cards_piano.append(generar_tarjeta_piano(clave, modo_acordes))


# ==========================================
# 6. PENTAGRAMA RÍTMICO Y EXPORTACIÓN
# ==========================================
def generar_partitura_pentagrama_ritmica(eventos_partitura, nombre_salida="partitura_ritmica.png"):
    """Dibuja un pentagrama con las notas/acordes tocados. La duración
    sostenida decide la figura: redonda, blanca, negra o corchea."""
    if len(eventos_partitura) == 0:
        print("No se registraron notas para generar el pentagrama.")
        return

    ancho_lienzo = 1000
    alto_sistema = 220                      # Alto de cada línea de pentagrama
    num_eventos = len(eventos_partitura)
    num_sistemas = math.ceil(max(1, num_eventos) / 8)   # 8 notas por pentagrama

    alto_total = 110 + num_sistemas * alto_sistema
    img_partitura = np.full((alto_total, ancho_lienzo, 3), 255, dtype=np.uint8)

    # Cabecera
    cv2.rectangle(img_partitura, (0, 0), (ancho_lienzo - 1, 65), (0, 0, 0), 2)
    cv2.putText(img_partitura, "PARTITURA MUSICAL EN PENTAGRAMA", (25, 42), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 0), 2)

    # Distancia vertical (px) de cada nota respecto a la 1ª línea del pentagrama
    # (clave de Sol: línea superior = Fa5, línea inferior = Mi4). 6 px = medio paso.
    MAPEO_OFFSET_NOTAS = {
        "FA5": 0,   "MI5": 6,   "RE5": 12,  "DO5": 18,
        "SI4": 24,  "LA4": 30,  "SOL4": 36, "FA4": 42,
        "MI4": 48,  "RE4": 54,  "DO4": 60,
        "DO": 60,   "RE": 54,   "MI": 48,   "FA": 42,
        "SOL": 36,  "LA": 30,   "SI": 24
    }

    # Notas de cada acorde (tríada fundamental) para dibujarlas apiladas
    MAPEO_ACORDE_FUNDAMENTAL = {
        "I (Do Mayor)":     ["DO4", "MI4", "SOL4"],
        "II (Re menor)":    ["RE4", "FA4", "LA4"],
        "III (Mi menor)":   ["MI4", "SOL4", "SI4"],
        "IV (Fa Mayor)":    ["FA4", "LA4", "DO5"],
        "V (Sol Mayor)":    ["SOL4", "SI4", "RE5"],
        "VI (La menor)":    ["LA4", "DO5", "MI5"],
        "VII (Si dim)":     ["SI4", "RE5", "FA5"]
    }

    idx_ev = 0  # Índice del evento que toca dibujar
    for s in range(num_sistemas):
        y_base_pentagrama = 140 + s * alto_sistema

        # 5 líneas del pentagrama separadas 12 px
        for l in range(5):
            yl = y_base_pentagrama + l * 12
            cv2.line(img_partitura, (60, yl), (ancho_lienzo - 40, yl), (0, 0, 0), 1)

        # Marca de clave (texto "SOL") y barra inicial
        cv2.putText(img_partitura, "SOL", (68, y_base_pentagrama + 32), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
        cv2.line(img_partitura, (115, y_base_pentagrama), (115, y_base_pentagrama + 48), (0, 0, 0), 2)

        paso_x = (ancho_lienzo - 210) // 8   # Separación horizontal entre notas
        for pos_sist in range(8):
            if idx_ev >= len(eventos_partitura):
                break

            clave, modo_ac, dur_s = eventos_partitura[idx_ev]
            x_nota = 150 + pos_sist * paso_x

            # Lista de notas a dibujar y texto bajo el pentagrama
            if modo_ac:
                notas_list = MAPEO_ACORDE_FUNDAMENTAL.get(clave, ["DO4", "MI4", "SOL4"])
                lbl_texto = ETIQUETA_CORTA_ACORDE.get(clave, clave)
            else:
                notas_list = [f"{clave}4" if f"{clave}4" in MAPEO_OFFSET_NOTAS else clave]
                lbl_texto = clave

            # Figura según la duración sostenida (en segundos)
            es_redonda = (dur_s >= 3.0)
            es_blanca = (1.5 <= dur_s < 3.0)
            es_negra = (0.75 <= dur_s < 1.5)
            es_corchea = (dur_s < 0.75)

            cabeza_llena = es_negra or es_corchea   # Negra/corchea: cabeza rellena
            tiene_plica = not es_redonda            # La redonda no lleva plica

            y_min_acorde = 999  # Nota más alta (menor y) para colocar la plica
            for n_nombre in notas_list:
                off_y = MAPEO_OFFSET_NOTAS.get(n_nombre, 36)
                yn = y_base_pentagrama + off_y
                y_min_acorde = min(y_min_acorde, yn)

                # Línea adicional para el Do4 (debajo del pentagrama)
                if off_y >= 60:
                    cv2.line(img_partitura, (x_nota - 9, y_base_pentagrama + 60), (x_nota + 9, y_base_pentagrama + 60), (0, 0, 0), 1)

                # Cabeza de nota: elipse rellena (-1) o sólo contorno (2)
                if cabeza_llena:
                    cv2.ellipse(img_partitura, (x_nota, yn), (6, 4), 30, 0, 360, (0, 0, 0), -1)
                else:
                    cv2.ellipse(img_partitura, (x_nota, yn), (6, 4), 30, 0, 360, (0, 0, 0), 2)

            # Plica (palito vertical) y corchete de la corchea
            if tiene_plica:
                x_plica = x_nota + 5
                y_top_plica = y_min_acorde - 28
                cv2.line(img_partitura, (x_plica, y_min_acorde), (x_plica, y_top_plica), (0, 0, 0), 2)

                if es_corchea:
                    cv2.line(img_partitura, (x_plica, y_top_plica), (x_plica + 8, y_top_plica + 10), (0, 0, 0), 2)

            # Nombre bajo el pentagrama
            cv2.putText(img_partitura, lbl_texto, (x_nota - 14, y_base_pentagrama + 72), cv2.FONT_HERSHEY_SIMPLEX, 0.44, (0, 0, 0), 1)

            idx_ev += 1

        # Barra final del pentagrama
        cv2.line(img_partitura, (ancho_lienzo - 40, y_base_pentagrama), (ancho_lienzo - 40, y_base_pentagrama + 48), (0, 0, 0), 2)

    cv2.imwrite(nombre_salida, img_partitura)
    mostrar_imagen_segura("Partitura Pentagrama", img_partitura)
    print(f"Partitura en pentagrama exportada con éxito en: {nombre_salida}")


def exportar_tablaturas(lista_tarjetas_fichas, lista_tarjetas_piano, eventos_partitura):
    """Exporta tres imágenes: tablatura de fichas, tablatura de piano y pentagrama."""
    if len(lista_tarjetas_fichas) == 0:
        print("No se registraron notas para exportar tablaturas.")
        return

    columnas = 4   # Tarjetas por fila

    # ---------- Tablatura de fichas (fondo negro) ----------
    filas_f = math.ceil(len(lista_tarjetas_fichas) / columnas)
    cards_f = list(lista_tarjetas_fichas)
    blank_f = np.zeros((160, 220, 3), dtype=np.uint8)
    # Rellena con tarjetas vacías hasta completar la última fila
    while len(cards_f) % columnas != 0:
        cards_f.append(blank_f)
    filas_img_f = [np.hstack(cards_f[f * columnas: (f + 1) * columnas]) for f in range(filas_f)]
    tab_fichas_body = np.vstack(filas_img_f)

    # CORRECCIÓN: shape es una tupla (alto, ancho, canales); el ancho es shape[1]
    ancho_f = tab_fichas_body.shape[1]
    enc_f = np.zeros((60, ancho_f, 3), dtype=np.uint8)
    cv2.putText(enc_f, "TABLATURA DE GESTOS / FICHAS DE COLOR", (20, 38), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
    tab_fichas_final = np.vstack([enc_f, tab_fichas_body])
    cv2.imwrite("tablatura_fichas.png", tab_fichas_final)
    mostrar_imagen_segura("Tablatura de Fichas", tab_fichas_final)

    # ---------- Tablatura de piano (fondo blanco) ----------
    filas_p = math.ceil(len(lista_tarjetas_piano) / columnas)
    cards_p = list(lista_tarjetas_piano)
    blank_p = np.full((210, 300, 3), 255, dtype=np.uint8)
    while len(cards_p) % columnas != 0:
        cards_p.append(blank_p)
    filas_img_p = [np.hstack(cards_p[f * columnas: (f + 1) * columnas]) for f in range(filas_p)]
    tab_piano_body = np.vstack(filas_img_p)

    ancho_p = tab_piano_body.shape[1]   # Misma corrección que arriba
    enc_p = np.full((65, ancho_p, 3), 255, dtype=np.uint8)
    cv2.rectangle(enc_p, (0, 0), (ancho_p - 1, 64), (0, 0, 0), 2)
    cv2.putText(enc_p, "TABLATURA DE DIBUJOS DE PIANO", (20, 42), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 0), 2)
    tab_piano_final = np.vstack([enc_p, tab_piano_body])
    cv2.imwrite("tablatura_piano.png", tab_piano_final)
    mostrar_imagen_segura("Tablatura de Piano (Fondo Blanco)", tab_piano_final)

    # ---------- Pentagrama ----------
    generar_partitura_pentagrama_ritmica(eventos_partitura)


# ==========================================
# 7. SISTEMA DE TUTORIAL CON CARGA DE LÁMINAS (lamina_1 a lamina_5)
# ==========================================
# Códigos de teclas para waitKeyEx. Con waitKey(0) & 0xFF las flechas de Windows
# se leen como 0 y no se podían distinguir, por eso se usa waitKeyEx.
TECLAS_IZQUIERDA = {2424832, 65361, 2, ord('a'), ord('A')}
TECLAS_DERECHA = {2555904, 65363, 3, 32, ord('d'), ord('D')}
TECLAS_SALIR_TUTORIAL = {13, 27, ord('q'), ord('Q')}


def mostrar_carrusel_tutorial(nombre_ventana):
    """Muestra las 5 láminas del tutorial. Si no hay imagen para alguna,
    dibuja una diapositiva de texto equivalente. Se navega con flechas/A/D/espacio
    y se sale con ENTER, ESC o Q."""
    ancho, alto = 800, 520
    diapositiva = 0
    total_diapositivas = 5
    nombres_laminas = ["lamina_1", "lamina_2", "lamina_3", "lamina_4", "lamina_5"]

    while True:
        nombre_base = nombres_laminas[diapositiva]
        img_cargada = cargar_imagen_lamina(nombre_base)

        if img_cargada is not None:
            # Hay imagen de lámina: se ajusta al tamaño de la ventana
            img = cv2.resize(img_cargada, (ancho, alto))
        else:
            # NOTA: cv2.putText (fuente Hershey) no dibuja tildes, ñ ni viñetas "•",
            # por eso los textos van sin acentos y con "-" como viñeta.
            img = np.full((alto, ancho, 3), 20, dtype=np.uint8)
            cv2.rectangle(img, (15, 15), (ancho - 15, alto - 15), (255, 255, 0), 2)
            cv2.rectangle(img, (20, 20), (ancho - 20, 70), (40, 40, 40), -1)
            cv2.putText(img, f"TUTORIAL SINTETIZADOR OPENCV ({diapositiva + 1}/{total_diapositivas})", (35, 52),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 0), 2)

            if diapositiva == 0:
                # Introducción y fichas
                cv2.putText(img, "1. TITULO E INTRODUCCION", (40, 115), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)
                cv2.putText(img, "Sintetizador Gestual de Audio por Vision Artificial", (40, 155), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (200, 200, 200), 1)

                fichas_info = [("ROSA", (255, 0, 255)), ("AZUL", (255, 0, 0)), ("VERDE", (0, 255, 0)), ("ROJA", (0, 0, 255))]
                for i, (nombre_f, col) in enumerate(fichas_info):
                    x = 80 + i * 170
                    cv2.circle(img, (x + 30, 230), 28, col, -1)
                    cv2.putText(img, nombre_f, (x + 10, 280), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

                cv2.putText(img, "- Al tapar o destapar fichas con las manos generas notas y acordes.", (40, 335), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (255, 255, 255), 1)
                cv2.putText(img, "- El sistema detecta automaticamente la combinacion gestual.", (40, 370), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (255, 255, 255), 1)

            elif diapositiva == 1:
                # Tabla de combinaciones
                cv2.putText(img, "2. COMBINACION DE ACORDES Y NOTAS", (40, 115), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)
                combinaciones = [
                    ("Todas visibles", "DO / DoM (I)"), ("Rosa tapado", "RE / Rem (II)"),
                    ("Azul tapado", "MI / Mim (III)"), ("Verde tapado", "FA / FaM (IV)"),
                    ("Rojo tapado", "SOL / SolM (V)"), ("Rosa + Azul tapados", "LA / Lam (VI)"),
                    ("Rosa + Verde tapados", "SI / Sidim (VII)")
                ]
                for idx, (gesto, sonido) in enumerate(combinaciones):
                    y = 160 + idx * 42
                    cv2.putText(img, f"- {gesto}:", (50, y), cv2.FONT_HERSHEY_SIMPLEX, 0.46, (200, 200, 200), 1)
                    cv2.putText(img, sonido, (320, y), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (255, 255, 0), 2)

            elif diapositiva == 2:
                # Modos e instrumentos
                cv2.putText(img, "3. MODOS E INSTRUMENTOS", (40, 115), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)
                cv2.rectangle(img, (40, 145), (760, 250), (35, 35, 35), -1)
                cv2.rectangle(img, (40, 145), (760, 250), (255, 255, 0), 1)
                cv2.putText(img, "CAMBIO DE MODO (TECLA 'M')", (55, 175), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 0), 2)
                cv2.putText(img, "- Modo Acordes: Triadas principales (DoM, Rem, Mim, FaM, SolM, Lam, Sidim).", (55, 205), cv2.FONT_HERSHEY_SIMPLEX, 0.44, (255, 255, 255), 1)
                cv2.putText(img, "- Modo Notas: Notas individuales en estado fundamental (Do, Re, Mi...).", (55, 230), cv2.FONT_HERSHEY_SIMPLEX, 0.44, (255, 255, 255), 1)

                cv2.rectangle(img, (40, 275), (760, 395), (35, 35, 35), -1)
                cv2.rectangle(img, (40, 275), (760, 395), (255, 255, 0), 1)
                cv2.putText(img, "CAMBIO DE INSTRUMENTO (TECLA 'I')", (55, 305), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 0), 2)
                cv2.putText(img, "- En Modo Acordes: Sintetizador -> Piano -> Orquesta.", (55, 335), cv2.FONT_HERSHEY_SIMPLEX, 0.44, (255, 255, 255), 1)
                cv2.putText(img, "- En Modo Notas: Sintetizador -> Piano -> Violonchelo.", (55, 365), cv2.FONT_HERSHEY_SIMPLEX, 0.44, (255, 255, 255), 1)

            elif diapositiva == 3:
                # Loopstation
                cv2.putText(img, "4. LOOPSTATION: GRABAR, REPRODUCIR Y BORRAR", (40, 115), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)
                cv2.putText(img, "- GRABAR (Tecla 'R'):", (40, 160), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2)
                cv2.putText(img, "  Pulsa R para armar. Espera al pulso del metronomo (60 BPM) y empieza a grabar.", (40, 185), cv2.FONT_HERSHEY_SIMPLEX, 0.43, (255, 255, 255), 1)
                cv2.putText(img, "- CUANTIZAR Y REPRODUCIR (Tecla 'R' de nuevo):", (40, 225), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2)
                cv2.putText(img, "  Pulsa R para cerrar. Ajusta la duracion exacta a pulsos y suena en bucle.", (40, 250), cv2.FONT_HERSHEY_SIMPLEX, 0.43, (255, 255, 255), 1)
                cv2.putText(img, "- BORRAR LOOP (Tecla 'C'):", (40, 315), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2)
                cv2.putText(img, "  Pulsa C (Clear) para detener la reproduccion y eliminar la grabacion.", (40, 340), cv2.FONT_HERSHEY_SIMPLEX, 0.43, (255, 255, 255), 1)

            elif diapositiva == 4:
                # Exportación y resumen de controles
                cv2.putText(img, "5. EXPORTACION Y CONTROLES", (40, 115), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)
                cv2.putText(img, "- TECLA 'P': Graba y exporta partitura y 2 tablaturas (Fichas y Piano).", (40, 165), cv2.FONT_HERSHEY_SIMPLEX, 0.46, (255, 255, 0), 1)
                cv2.putText(img, "- TECLA 'T': Activa o desactiva el metronomo visual y sonoro a 60 BPM.", (40, 205), cv2.FONT_HERSHEY_SIMPLEX, 0.46, (255, 255, 255), 1)
                cv2.putText(img, "- TECLA 'I': Cambia de instrumento.", (40, 245), cv2.FONT_HERSHEY_SIMPLEX, 0.46, (255, 255, 255), 1)
                cv2.putText(img, "- TECLA 'M': Alterna entre Acordes y Notas.", (40, 285), cv2.FONT_HERSHEY_SIMPLEX, 0.46, (255, 255, 255), 1)
                cv2.putText(img, "- TECLA 'H': Abre este tutorial en cualquier momento.", (40, 325), cv2.FONT_HERSHEY_SIMPLEX, 0.46, (255, 255, 0), 1)

            # Barra inferior con las instrucciones de navegación
            cv2.rectangle(img, (20, alto - 55), (ancho - 20, alto - 15), (30, 30, 30), -1)
            cv2.putText(img, "[FLECHAS / ESPACIO: Navegar]  |  [ENTER / ESC: Empezar Programa]", (45, alto - 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.46, (255, 255, 255), 1)

        mostrar_imagen_segura(nombre_ventana, img)
        key = cv2.waitKeyEx(0)   # Espera una tecla (sin límite de tiempo)

        if key in TECLAS_SALIR_TUTORIAL:
            break
        elif key in TECLAS_IZQUIERDA:
            diapositiva = (diapositiva - 1) % total_diapositivas   # Anterior (circular)
        elif key in TECLAS_DERECHA:
            diapositiva = (diapositiva + 1) % total_diapositivas   # Siguiente (circular)


# ==========================================
# 8. BUCLE PRINCIPAL (CON LOOP SINCRONIZADO Y CUANTIZADO)
# ==========================================
def main():
    # Prepara el audio (genera/carga todos los sonidos)
    if WINSOUND_DISPONIBLE or PYGAME_DISPONIBLE:
        pregenerar_sonidos()

    # Abre la cámara 1; si no existe, prueba con la 0 (cámara integrada)
    camara = cv2.VideoCapture(1)
    if not camara.isOpened():
        camara.release()
        camara = cv2.VideoCapture(0)

    if not camara.isOpened():
        print("Error: No es posible abrir la cámara web.")
        return

    # Ventana con un trackbar para ajustar el brillo de la imagen
    nombre_ventana = "Gesture Synth OpenCV + Loopstation"
    try:
        cv2.namedWindow(nombre_ventana)
        cv2.createTrackbar("Ajuste Brillo", nombre_ventana, 100, 200, actualizar_brillo_trackbar)
    except Exception:
        pass

    # Tutorial al lanzar el programa
    mostrar_carrusel_tutorial(nombre_ventana)

    # Instrumentos disponibles en cada modo
    insts_acordes = ["SINTETIZADOR", "PIANO", "ORQUESTA"]
    insts_notas = ["SINTETIZADOR", "PIANO", "VIOLONCHELO"]

    modo_acordes = True
    instrumentos_actuales = insts_acordes
    idx_instrumento = 0
    instrumento_actual = instrumentos_actuales[idx_instrumento]

    # ---- Estado de la nota detectada ----
    clave_actual = "SILENCIO"       # Nota/acorde confirmada que suena ahora
    clave_candidata = "SILENCIO"    # Nota detectada que aún debe estabilizarse
    t_inicio_candidata = time.time()

    TIEMPO_MINIMO_ESTABILIDAD = 0.18   # Segundos que debe mantenerse un gesto para aceptarlo
    UMBRAL_MINIMO_RECORD_NOTA = 0.25   # Notas más cortas no se incluyen en la partitura

    clave_confirmada_anterior = "SILENCIO"   # Última nota confirmada (para medir su duración)
    t_inicio_nota_confirmada = 0.0

    # ---- Metrónomo a 60 BPM (1 pulso por segundo) ----
    metronomo_activo = True
    bpm = 60
    intervalo_beat = 60.0 / bpm
    t_ultimo_beat = time.time()
    beat_contador = 1

    # ---- Loopstation ----
    # Estados: 'IDLE' (inactiva), 'ARMED_REC' (armada, espera pulso),
    #          'RECORDING' (grabando), 'PLAYING' (reproduciendo en bucle)
    estado_loop = 'IDLE'
    tiempo_inicio_grabacion = 0
    duracion_loop = 0
    eventos_grabados = []        # (tiempo_relativo, clave, modo_acordes, instrumento)
    tiempo_inicio_reproduccion = 0
    indice_evento_loop = -1

    # ---- Grabación de partitura (tecla P) ----
    grabando_tablatura = False
    lista_cards_fichas = []
    lista_cards_piano = []
    eventos_partitura_ritmica = []

    while True:
        ret, frame = camara.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)   # Espejo, para que sea natural al mover las manos
        t_actual = time.time()

        # ---------- Metrónomo ----------
        destello_metronomo = False
        if metronomo_activo:
            if t_actual - t_ultimo_beat >= intervalo_beat:
                t_ultimo_beat = t_actual
                beat_contador += 1
                destello_metronomo = True
                sonar_click_metronomo()

                # Si la loopstation está armada, empieza a grabar EXACTAMENTE en este pulso
                if estado_loop == 'ARMED_REC':
                    estado_loop = 'RECORDING'
                    tiempo_inicio_grabacion = t_actual
                    # Primer evento: lo que ya estuviera sonando en ese instante
                    eventos_grabados = [(0.0, clave_actual, modo_acordes, instrumento_actual)]
                    print("Pulso marcado: Grabación de Loop iniciada en sincronía.")

        # ---------- Detección de color ----------
        frame_ajustado = ajustar_brillo_frame(frame, incremento_brillo)
        frame_suavizado = cv2.blur(frame_ajustado, (5, 5))       # Reduce ruido
        hsv = cv2.cvtColor(frame_suavizado, cv2.COLOR_BGR2HSV)

        rosa_visible, _, c_rosa = evaluar_visibilidad(hsv, rosa_bajo, rosa_alto)
        azul_visible, _, c_azul = evaluar_visibilidad(hsv, azul_bajo, azul_alto)
        verde_visible, _, c_verde = evaluar_visibilidad(hsv, verde_bajo, verde_alto)
        roja_visible, _, c_rojo = evaluar_visibilidad(hsv, rojo_bajo1, rojo_alto1, rojo_bajo2, rojo_alto2)

        rosa_tapado = not rosa_visible
        azul_tapado = not azul_visible
        verde_tapado = not verde_visible
        rojo_tapado = not roja_visible

        # ---------- Combinación de fichas -> nota o acorde ----------
        if rosa_visible and azul_visible and verde_visible and roja_visible:
            clave_detectada_instante = "I (Do Mayor)" if modo_acordes else "DO"
        elif rosa_tapado and azul_tapado and verde_visible and roja_visible:
            clave_detectada_instante = "VI (La menor)" if modo_acordes else "LA"
        elif rosa_tapado and verde_tapado and azul_visible and roja_visible:
            clave_detectada_instante = "VII (Si dim)" if modo_acordes else "SI"
        elif rosa_tapado and azul_visible and verde_visible and roja_visible:
            clave_detectada_instante = "II (Re menor)" if modo_acordes else "RE"
        elif azul_tapado and rosa_visible and verde_visible and roja_visible:
            clave_detectada_instante = "III (Mi menor)" if modo_acordes else "MI"
        elif verde_tapado and rosa_visible and azul_visible and roja_visible:
            clave_detectada_instante = "IV (Fa Mayor)" if modo_acordes else "FA"
        elif rojo_tapado and rosa_visible and azul_visible and verde_visible:
            clave_detectada_instante = "V (Sol Mayor)" if modo_acordes else "SOL"
        else:
            clave_detectada_instante = "SILENCIO"   # Combinación no reconocida

        # ---------- Filtro anti-rebote ----------
        # Un gesto sólo se acepta si se mantiene TIEMPO_MINIMO_ESTABILIDAD segundos
        if clave_detectada_instante != clave_candidata:
            clave_candidata = clave_detectada_instante
            t_inicio_candidata = t_actual
        else:
            if (t_actual - t_inicio_candidata >= TIEMPO_MINIMO_ESTABILIDAD) and (clave_candidata != clave_actual):
                clave_actual = clave_candidata

                if clave_actual != clave_confirmada_anterior:
                    # Si se está grabando la partitura, cierra la nota anterior con su duración
                    if grabando_tablatura and clave_confirmada_anterior != "SILENCIO":
                        dur_sostenida = t_actual - t_inicio_nota_confirmada
                        if dur_sostenida >= UMBRAL_MINIMO_RECORD_NOTA:
                            registrar_nota_partitura(clave_confirmada_anterior, modo_acordes, dur_sostenida,
                                                     eventos_partitura_ritmica, lista_cards_fichas, lista_cards_piano)

                    t_inicio_nota_confirmada = t_actual
                    reproducir_sonido_vivo(clave_actual, modo_acordes, instrumento_actual)

                    # Si se está grabando el loop, anota el cambio de nota con su tiempo relativo
                    if estado_loop == 'RECORDING':
                        t_rel = t_actual - tiempo_inicio_grabacion
                        eventos_grabados.append((t_rel, clave_actual, modo_acordes, instrumento_actual))

                    clave_confirmada_anterior = clave_actual

        # ---------- Reproducción del loop sincronizado ----------
        if estado_loop == 'PLAYING' and duracion_loop > 0 and len(eventos_grabados) > 0:
            # Posición actual dentro del bucle (vuelve a 0 al llegar al final)
            t_en_loop = (t_actual - tiempo_inicio_reproduccion) % duracion_loop

            # Busca el último evento cuyo instante ya ha pasado
            idx_evento = 0
            for i, (t_ev, _, _, _) in enumerate(eventos_grabados):
                if t_en_loop >= t_ev:
                    idx_evento = i
                else:
                    break

            # Sólo cambia el sonido cuando se pasa a otro evento
            if idx_evento != indice_evento_loop:
                _, clave_loop, modo_loop_ev, inst_loop_ev = eventos_grabados[idx_evento]
                reproducir_sonido_loop(clave_loop, modo_loop_ev, inst_loop_ev)
                indice_evento_loop = idx_evento

        # ---------- Dibujo sobre el vídeo ----------
        fichas = [
            ("ROSA", rosa_visible, c_rosa, (255, 0, 255)),
            ("AZUL", azul_visible, c_azul, (255, 0, 0)),
            ("VERDE", verde_visible, c_verde, (0, 255, 0)),
            ("ROJA", roja_visible, c_rojo, (0, 0, 255))
        ]

        # Recuadro y nombre alrededor de cada ficha visible
        for nombre, visible, contorno, color_bgr in fichas:
            if visible and contorno is not None:
                x, y, w, h = cv2.boundingRect(contorno)
                cv2.rectangle(frame_ajustado, (x, y), (x + w, y + h), color_bgr, 2)
                cv2.putText(frame_ajustado, nombre, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color_bgr, 2)

        # Panel de información
        cv2.rectangle(frame_ajustado, (20, 15), (610, 150), (0, 0, 0), -1)
        cv2.rectangle(frame_ajustado, (20, 15), (610, 150), (255, 255, 255), 2)

        modo_str = "ACORDES" if modo_acordes else "NOTAS INDIVIDUALES"
        cv2.putText(frame_ajustado, f"INSTRUMENTO: {instrumento_actual} | MODO: {modo_str}", (30, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 1)
        cv2.putText(frame_ajustado, f"EN VIVO: {clave_actual}", (30, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

        # Indicador visual del metrónomo (se ilumina en cada pulso)
        if metronomo_activo:
            col_met = (0, 255, 255) if destello_metronomo else (40, 180, 40)
            cv2.rectangle(frame_ajustado, (420, 20), (600, 85), col_met, -1)
            cv2.rectangle(frame_ajustado, (420, 20), (600, 85), (255, 255, 255), 2)
            cv2.circle(frame_ajustado, (445, 52), 16, (0, 0, 0) if destello_metronomo else (255, 255, 255), -1)
            cv2.putText(frame_ajustado, "METRONOMO 60BPM", (470, 42), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 2)
            cv2.putText(frame_ajustado, f"PULSO {beat_contador}", (470, 68), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (0, 0, 0), 2)

        # Texto y color según el estado de la loopstation
        if estado_loop == 'IDLE':
            loop_str = "LOOPSTATION: Inactivo (Pulsa 'R' para armar)"
            color_loop = (200, 200, 200)
        elif estado_loop == 'ARMED_REC':
            loop_str = "ARMADO: Esperando proximo pulso de metronomo..."
            color_loop = (0, 165, 255)
        elif estado_loop == 'RECORDING':
            tiempo_rec = t_actual - tiempo_inicio_grabacion
            loop_str = f"GRABANDO LOOP CUANTIZADO ({tiempo_rec:.1f}s) - Pulsa 'R' para cerrar"
            color_loop = (0, 0, 255)
        else:  # 'PLAYING'
            loop_str = f"REPRODUCIENDO LOOP ({duracion_loop:.1f}s en sync) - Pulsa 'C' para borrar"
            color_loop = (0, 255, 0)

        cv2.putText(frame_ajustado, loop_str, (30, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.43, color_loop, 2)

        # Estado de la grabación de partitura (sin tildes: Hershey no las dibuja)
        tab_str = f"PARTITURA: GRABANDO ({len(lista_cards_fichas)} notas)" if grabando_tablatura else "PARTITURA: Pulsa 'P' para exportar 3 laminas"
        color_tab = (0, 0, 255) if grabando_tablatura else (180, 180, 180)
        cv2.putText(frame_ajustado, tab_str, (30, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.42, color_tab, 1)

        cv2.putText(frame_ajustado, "['T': Metronomo | 'I': Inst | 'M': Modo | 'R': Loop Sync | 'P': Partitura | 'C': Borrar | 'H': Ayuda]", (30, 140), cv2.FONT_HERSHEY_SIMPLEX, 0.34, (180, 180, 180), 1)

        mostrar_imagen_segura(nombre_ventana, frame_ajustado)

        # ---------- Teclado ----------
        key = cv2.waitKey(1) & 0xFF

        if key in [ord('t'), ord('T')]:
            # Activa/desactiva el metrónomo y reinicia su cuenta
            metronomo_activo = not metronomo_activo
            beat_contador = 1
            t_ultimo_beat = time.time()
            if metronomo_activo:
                sonar_click_metronomo()
            print(f"Metrónomo 60 BPM: {'ACTIVADO' if metronomo_activo else 'DESACTIVADO'}")

        elif key in [ord('i'), ord('I')]:
            # Siguiente instrumento (circular) dentro del modo actual
            idx_instrumento = (idx_instrumento + 1) % len(instrumentos_actuales)
            instrumento_actual = instrumentos_actuales[idx_instrumento]
            reproducir_sonido_vivo(clave_actual, modo_acordes, instrumento_actual)
            # Si se está grabando un loop, el cambio de timbre también queda registrado
            if estado_loop == 'RECORDING':
                eventos_grabados.append((time.time() - tiempo_inicio_grabacion, clave_actual, modo_acordes, instrumento_actual))
            print(f"Instrumento cambiado a: {instrumento_actual}")

        elif key in [ord('m'), ord('M')]:
            # Antes de cambiar de modo, cierra la nota que se estuviera grabando en la partitura
            if grabando_tablatura and clave_confirmada_anterior != "SILENCIO":
                dur_sostenida = time.time() - t_inicio_nota_confirmada
                if dur_sostenida >= UMBRAL_MINIMO_RECORD_NOTA:
                    registrar_nota_partitura(clave_confirmada_anterior, modo_acordes, dur_sostenida,
                                             eventos_partitura_ritmica, lista_cards_fichas, lista_cards_piano)

            # Cambia de modo y vuelve al primer instrumento de ese modo
            modo_acordes = not modo_acordes
            instrumentos_actuales = insts_acordes if modo_acordes else insts_notas
            idx_instrumento = 0
            instrumento_actual = instrumentos_actuales[idx_instrumento]

            # Las claves de un modo no existen en el otro: se corta el sonido y se
            # reinicia la detección. (El código original ponía None aquí, lo que podía
            # provocar un error al grabar la partitura y dejaba sonando la nota vieja.)
            detener_audio_vivo()
            clave_actual = "SILENCIO"
            clave_candidata = "SILENCIO"
            clave_confirmada_anterior = "SILENCIO"
            t_inicio_candidata = time.time()
            print(f"Modo cambiado a: {'ACORDES' if modo_acordes else 'NOTAS'} | Instrumento: {instrumento_actual}")

        elif key in [ord('p'), ord('P')]:
            if not grabando_tablatura:
                # Primera pulsación: empieza a registrar notas para la partitura
                grabando_tablatura = True
                t_inicio_nota_confirmada = time.time()   # La nota en curso cuenta desde ahora
                lista_cards_fichas.clear()
                lista_cards_piano.clear()
                eventos_partitura_ritmica.clear()
                print("Iniciada grabación de partitura y tablaturas...")
            else:
                # Segunda pulsación: cierra la última nota y exporta las 3 láminas
                grabando_tablatura = False
                if clave_actual != "SILENCIO":
                    dur_sostenida = time.time() - t_inicio_nota_confirmada
                    if dur_sostenida >= UMBRAL_MINIMO_RECORD_NOTA:
                        registrar_nota_partitura(clave_actual, modo_acordes, dur_sostenida,
                                                 eventos_partitura_ritmica, lista_cards_fichas, lista_cards_piano)
                print("Exportando 3 láminas (Fichas, Piano y Pentagrama)...")
                exportar_tablaturas(lista_cards_fichas, lista_cards_piano, eventos_partitura_ritmica)

        elif key in [ord('r'), ord('R')]:
            ahora = time.time()
            if estado_loop == 'IDLE':
                # Arma la grabación (activa el metrónomo, que marca el inicio)
                metronomo_activo = True
                estado_loop = 'ARMED_REC'
                print("Loopstation armada: La grabación se iniciará en el próximo pulso de metrónomo.")
            elif estado_loop == 'ARMED_REC':
                # Segunda pulsación antes de empezar: cancela el armado
                estado_loop = 'IDLE'
                print("Armado de Loop cancelado.")
            elif estado_loop == 'RECORDING':
                # Cierra la grabación y CUANTIZA: redondea la duración al número
                # entero de pulsos más cercano para que el bucle no pierda el ritmo
                duracion_bruta = ahora - tiempo_inicio_grabacion
                num_beats = max(1, round(duracion_bruta / intervalo_beat))
                duracion_loop = float(num_beats * intervalo_beat)

                # Descarta eventos que caigan fuera de la duración cuantizada
                eventos_grabados = [e for e in eventos_grabados if e[0] < duracion_loop]

                if len(eventos_grabados) > 0:
                    estado_loop = 'PLAYING'
                    tiempo_inicio_reproduccion = t_ultimo_beat   # Alineado al último pulso
                    indice_evento_loop = -1
                    print(f"Loop grabado y cuantizado a {num_beats} pulso(s) ({duracion_loop:.1f}s). Reproduciendo en sync...")
                else:
                    estado_loop = 'IDLE'
            elif estado_loop == 'PLAYING':
                # R durante la reproducción: se vuelve a armar para grabar otro loop
                estado_loop = 'ARMED_REC'
                detener_audio_loop()
                print("Reiniciando grabación de nuevo Loop (esperando pulso)...")

        elif key in [ord('c'), ord('C')]:
            # Borra por completo el loop y deja la loopstation limpia
            estado_loop = 'IDLE'
            eventos_grabados = []
            duracion_loop = 0
            detener_audio_loop()
            print("Loop borrado.")

        elif key in [ord('h'), ord('H')]:
            # Vuelve a abrir el tutorial (el texto de ayuda de la lámina 5 lo prometía,
            # pero esta tecla no estaba implementada)
            mostrar_carrusel_tutorial(nombre_ventana)
            t_ultimo_beat = time.time()   # Evita un pulso "atrasado" tras la pausa

        elif key in [ord('q'), ord('Q'), 27]:
            break   # Salir

    # ---------- Limpieza al salir ----------
    detener_audio_vivo()
    detener_audio_loop()
    if WINSOUND_DISPONIBLE:
        try:
            winsound.PlaySound(None, winsound.SND_PURGE)
        except Exception:
            pass

    camara.release()
    try:
        cv2.destroyAllWindows()
    except Exception:
        pass


if __name__ == '__main__':
    main()