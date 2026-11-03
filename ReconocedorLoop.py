"""
===============================================================================
 SINTETIZADOR GESTUAL OPENCV + LOOPSTATION SINCRONIZADA
===============================================================================
 QUÉ HACE ESTE PROGRAMA
   1. La cámara web graba cuatro fichas de colores (rosa, azul, verde y roja).
   2. OpenCV detecta qué fichas se ven y cuáles están tapadas por tu mano.
   3. Cada combinación de fichas visibles/tapadas es una nota (DO, RE, MI...) o
      un acorde (DoM, Rem, Mim...), que se reproduce como sonido.
   4. Un metrónomo a 60 BPM sirve de reloj para grabar loops sincronizados.
   5. Se puede exportar lo tocado como partitura y tablaturas (carpeta tabs/).
 
 CARPETAS (junto a este script)
   imagenes/  -> lamina_1 ... lamina_5 (.png/.jpg) y tab_*.png de piano
   audios/    -> cello_*.wav, piano_*.wav, orquesta_*.wav (muestras reales).
                 Los sonidos que se generan solos también se guardan aquí.
   tabs/      -> partituras y tablaturas exportadas con la tecla P
 
 CONTROLES
   T -> activa/desactiva el SONIDO del metrónomo (empieza desactivado)
   I -> cambiar instrumento          M -> modo acordes / notas
   R -> loopstation (grabar / cerrar y cuantizar)
   C -> borrar loop                  P -> grabar y exportar partitura + tablaturas
   H -> abrir/cerrar tutorial        Q o ESC -> salir
        (ESC cierra el tutorial si está abierto)
 
 TUTORIAL
   Se dibuja SOBRE la imagen de la cámara (misma ventana, más pequeño).
   Se cierra con ESC, ENTER, H o haciendo clic en la [X] roja.
   Se navega con las flechas, A/D, espacio o los botones < >.
 
-------------------------------------------------------------------------------
 GUÍA OPENCV (cv2) - las funciones que usa este programa
-------------------------------------------------------------------------------
 Ideas básicas antes de empezar:
   * Una imagen en OpenCV es un array de NumPy con forma (alto, ancho, 3).
     Cada píxel guarda 3 números de 0 a 255 en orden BGR (Azul, Verde, Rojo),
     No es RGB - Por eso el rojo puro se escribe (0, 0, 255).
   * Las coordenadas son (x, y) con el origen (0, 0) ARRIBA A LA IZQUIERDA:
     x crece hacia la derecha e y crece hacia ABAJO.
   * Casi todas las funciones de dibujo modifican la imagen "en sitio" (la
     misma variable que les pasas queda dibujada).
   * grosor = -1 significa "rellenar la figura". cv2.LINE_AA = bordes suaves.
 
 CÁMARA Y VENTANA
   cv2.VideoCapture(n)       Abre la cámara número n (0 = la primera).
   camara.read()             Captura un fotograma: devuelve (ok, imagen).
   cv2.flip(img, 1)          Voltea la imagen en horizontal (efecto espejo).
   cv2.namedWindow(nombre)   Crea la ventana donde se mostrará el vídeo.
   cv2.createTrackbar(...)   Añade un deslizador a la ventana (aquí, el brillo).
   cv2.setMouseCallback(...) Registra una función que se llama con cada evento
                             del ratón (clics, movimiento) en esa ventana.
   cv2.imshow(nombre, img)   Muestra una imagen en la ventana.
   cv2.waitKeyEx(ms)         Espera una tecla ms milisegundos. IMPRESCINDIBLE:
                             sin ella OpenCV no refresca la ventana. Devuelve
                             el código de la tecla (o -1 si no hubo ninguna).
                             La versión "Ex" también reconoce las flechas.
   cv2.destroyAllWindows()   Cierra todas las ventanas de OpenCV.
 
 DETECCIÓN DE COLOR (visión artificial)
   cv2.blur(img, (5, 5))     Desenfoque: promedia cada píxel con sus vecinos
                             para quitar ruido de la cámara.
   cv2.cvtColor(img, ...)    Convierte entre espacios de color. Aquí BGR->HSV.
                             HSV separa el color (H, matiz) de la intensidad
                             (S saturación, V brillo): detectar "rojo" en HSV
                             es mucho más estable que en BGR frente a la luz.
   cv2.split / cv2.merge     Separa una imagen en sus canales (H, S, V) y los
                             vuelve a unir.
   cv2.inRange(hsv, bajo, alto)  Crea una MÁSCARA: imagen en blanco/negro donde
                             blanco = píxeles cuyo color está dentro del rango.
   cv2.bitwise_or(a, b)      Une dos máscaras (blanco si lo es en cualquiera).
   cv2.erode / cv2.dilate    Operaciones morfológicas. Erosionar encoge las
                             manchas blancas (borra puntitos de ruido) y
                             dilatar las vuelve a agrandar.
   cv2.findContours(...)     Encuentra los contornos de las manchas blancas.
   cv2.contourArea(c)        Área (en píxeles) encerrada por un contorno.
   cv2.boundingRect(c)       Rectángulo (x, y, ancho, alto) que envuelve un
                             contorno.
 
 DIBUJO (HUD, tarjetas y partitura)
   cv2.rectangle  cv2.circle  cv2.line  cv2.ellipse  cv2.polylines
   cv2.fillPoly   Figuras geométricas (rectángulo, círculo, línea, elipse,
                  polilínea abierta/cerrada y polígono relleno).
   cv2.putText    Escribe texto (fuente Hershey: no dibuja tildes ni ñ).
   cv2.getTextSize  Mide cuánto ocupará un texto, para centrarlo o ajustarlo.
   cv2.addWeighted  Mezcla dos imágenes con transparencia:
                    resultado = a*alfa + b*(1-alfa). Sirve para paneles
                    translúcidos y efectos de resplandor ("glow") neón.
 
 IMÁGENES EN DISCO
   cv2.imdecode / cv2.imencode  Leer / guardar imágenes desde/hacia bytes en
                    memoria (funciona con rutas con tildes en Windows, a
                    diferencia de cv2.imread / cv2.imwrite).
   cv2.resize       Cambia el tamaño de una imagen.
===============================================================================
"""
 
# -----------------------------------------------------------------------------
# LIBRERÍAS Y PARA QUÉ SE USA CADA UNA
# -----------------------------------------------------------------------------
import cv2          # OpenCV: captura de cámara, detección de color, dibujo del HUD,
                    #   ventanas, teclado y ratón. Es el corazón del proyecto.
import numpy as np  # NumPy: cálculo con arrays. Las imágenes de OpenCV SON arrays
                    #   de NumPy, y también sintetizamos el audio con él (rápido).
import struct       # Convierte números a bytes. Se usa para escribir a mano la
                    #   cabecera de los archivos .wav que generamos.
import math         # Funciones matemáticas (ceil para el nº de pentagramas,
                    #   exp para el efecto de resplandor).
import time         # time.time() = reloj en segundos: metrónomo, duración de
                    #   las notas, estabilidad del gesto y strftime para nombres.
import os           # Rutas y carpetas (os.path.join, os.makedirs, os.path.exists).
 
# Librerías de audio (ambas OPCIONALES: si no están instaladas el programa sigue,
# simplemente sin ese tipo de sonido).
try:
    import pygame   # pygame.mixer: mezclador de audio con VARIOS CANALES a la
                    #   vez (uno para lo que tocas, otro para el loop, el click...).
    PYGAME_DISPONIBLE = True
except ImportError:
    PYGAME_DISPONIBLE = False
 
try:
    import winsound  # Sólo Windows: sonido básico del sistema (alternativa simple
                     #   a pygame, pero con un único canal).
    WINSOUND_DISPONIBLE = True
except ImportError:
    WINSOUND_DISPONIBLE = False
 
# ***************************************************************
# 0. CARPETAS 
# ***************************************************************
# __file__ es la ruta de este script; así las carpetas siempre se buscan junto a él,
# se ejecute desde donde se ejecute.
DIRECTORIO_SCRIPT = os.path.dirname(os.path.abspath(__file__))
DIR_IMAGENES = os.path.join(DIRECTORIO_SCRIPT, "imagenes")
DIR_AUDIOS = os.path.join(DIRECTORIO_SCRIPT, "audios")
DIR_TABS = os.path.join(DIRECTORIO_SCRIPT, "tabs")
 
FUENTE = cv2.FONT_HERSHEY_SIMPLEX  # Fuente de OpenCV usada en todos los textos
TIEMPO_MINIMO_ESTABILIDAD = 0.18   # Segundos que debe mantenerse un gesto para aceptarlo
UMBRAL_MINIMO_RECORD_NOTA = 0.25   # Las notas más cortas no entran en la partitura
UMBRAL_AREA_MINIMA = 3000          # Área mínima (px) para considerar una ficha visible
 
# ---- Paleta del HUD (tema azul/cian, igual que las láminas del tutorial) ----
# Recuerda: OpenCV usa orden BGR, así que (B, G, R).
COLOR_FONDO = (38, 20, 11)         # Azul marino   #0B1426
COLOR_CIAN = (232, 219, 127)       # Cian UI       #7FDBE8
COLOR_BLANCO = (250, 248, 242)     # Blanco azulado
COLOR_AMARILLO = (0, 230, 255)     # Amarillo neón #FFE600
COLOR_APAGADO = (140, 110, 70)     # Azul/cian oscuro para textos inactivos
COLOR_ROJO = (70, 70, 255)         # Rojo claro (grabando)
COLOR_VERDE = (130, 230, 100)      # Verde claro (reproduciendo)
 
 
def crear_carpetas():
    """Crea las carpetas de trabajo si todavía no existen."""
    # exist_ok=True evita el error si la carpeta ya estaba creada
    for carpeta in (DIR_IMAGENES, DIR_AUDIOS, DIR_TABS):
        os.makedirs(carpeta, exist_ok=True)
 
 
# ***************************************************************
# 1. FRECUENCIAS Y MAPEOS 
# ***************************************************************
# Cada nota es una frecuencia en Hz (vibraciones por segundo). Do4 = 261,63 Hz.
NOTAS_SOLO = {
    "DO": [261.63], "RE": [293.66], "MI": [329.63], "FA": [349.23],
    "SOL": [392.00], "LA": [440.00], "SI": [493.88]
}
 
# Tríadas en estado fundamental: tres frecuencias que suenan a la vez
ACORDES = {
    "I (Do Mayor)":   [261.63, 329.63, 392.00],
    "II (Re menor)":  [293.66, 349.23, 440.00],
    "III (Mi menor)": [329.63, 392.00, 493.88],
    "IV (Fa Mayor)":  [349.23, 440.00, 523.25],
    "V (Sol Mayor)":  [392.00, 493.88, 587.33],
    "VI (La menor)":  [440.00, 523.25, 659.25],
    "VII (Si dim)":   [493.88, 587.33, 698.46]
}
 
# Acorde -> etiqueta para nombres de archivo
MAPEO_CLAVE_ACORDE_ARCHIVO = {
    "I (Do Mayor)": "I_DoM", "II (Re menor)": "II_Rem", "III (Mi menor)": "III_Mim",
    "IV (Fa Mayor)": "IV_FaM", "V (Sol Mayor)": "V_SolM", "VI (La menor)": "VI_Lam",
    "VII (Si dim)": "VII_Sidim"
}
 
# Acorde -> etiqueta corta para pantalla
ETIQUETA_CORTA_ACORDE = {
    "I (Do Mayor)": "DoM", "II (Re menor)": "Rem", "III (Mi menor)": "Mim",
    "IV (Fa Mayor)": "FaM", "V (Sol Mayor)": "SolM", "VI (La menor)": "Lam",
    "VII (Si dim)": "Sidim"
}
 
# Acorde <-> nota sobre la que se construye
ACORDE_A_NOTA = {
    "I (Do Mayor)": "DO", "II (Re menor)": "RE", "III (Mi menor)": "MI",
    "IV (Fa Mayor)": "FA", "V (Sol Mayor)": "SOL", "VI (La menor)": "LA",
    "VII (Si dim)": "SI"
}
NOTA_A_ACORDE = {nota: acorde for acorde, nota in ACORDE_A_NOTA.items()}
 
# Nota -> etiqueta de archivo de acorde (para buscar su imagen de piano)
NOTA_A_TAG_ACORDE = {n: MAPEO_CLAVE_ACORDE_ARCHIVO[a] for n, a in NOTA_A_ACORDE.items()}
 
# Fichas VISIBLES para cada nota, en el orden (rosa, azul, verde, roja)
VISIBILIDAD_POR_NOTA = {
    "DO":  (True,  True,  True,  True),
    "RE":  (False, True,  True,  True),
    "MI":  (True,  False, True,  True),
    "FA":  (True,  True,  False, True),
    "SOL": (True,  True,  True,  False),
    "LA":  (False, False, True,  True),
    "SI":  (False, True,  False, True)
}
# Tabla inversa: combinación de fichas visibles -> nota (evita una cadena de if/elif)
NOTA_POR_PATRON = {patron: nota for nota, patron in VISIBILIDAD_POR_NOTA.items()}
 
INSTRUMENTOS = {
    True:  ["SINTETIZADOR", "PIANO", "ORQUESTA"],      # Modo acordes
    False: ["SINTETIZADOR", "PIANO", "VIOLONCHELO"]    # Modo notas
}
 
# ***************************************************************
# 2. SÍNTESIS Y CARGA DE AUDIO
# ***************************************************************
RUTAS_AUDIO = {i: {"NOTAS": {}, "ACORDES": {}} for i in ("SINTETIZADOR", "PIANO", "VIOLONCHELO", "ORQUESTA")}
SONIDOS_PYGAME = {i: {"NOTAS": {}, "ACORDES": {}} for i in ("SINTETIZADOR", "PIANO", "VIOLONCHELO", "ORQUESTA")}
SONIDO_METRONOMO = None
CANALES = {"vivo": None, "loop": None}   # Canal 1 = lo que tocas, canal 0 = loop grabado
 
# Catálogo: (instrumento, tipo, frecuencias, nombre de archivo que se busca en /audios)
CATALOGO_AUDIO = [
    ("SINTETIZADOR", "NOTAS",   NOTAS_SOLO, lambda c: f"synth_nota_{c}.wav"),
    ("SINTETIZADOR", "ACORDES", ACORDES,    lambda c: f"synth_acorde_{MAPEO_CLAVE_ACORDE_ARCHIVO[c]}.wav"),
    ("VIOLONCHELO",  "NOTAS",   NOTAS_SOLO, lambda c: f"cello_{c}.wav"),
    ("PIANO",        "NOTAS",   NOTAS_SOLO, lambda c: f"piano_nota_{c}.wav"),
    ("PIANO",        "ACORDES", ACORDES,    lambda c: f"piano_acorde_{MAPEO_CLAVE_ACORDE_ARCHIVO[c]}.wav"),
    ("ORQUESTA",     "ACORDES", ACORDES,    lambda c: f"orquesta_acorde_{MAPEO_CLAVE_ACORDE_ARCHIVO[c]}.wav"),
]
 
 
def generar_bytes_wav(freqs, duracion=1.0, sr=44100, estilo="SYNTH"):
    """Genera en memoria un WAV mono de 16 bits con una o varias frecuencias a la vez.
 
    freqs    -> lista de frecuencias en Hz (una nota = 1, un acorde = 3)
    duracion -> segundos de audio
    sr       -> "sample rate": muestras por segundo (44100 = calidad CD)
    estilo   -> "METRONOMO", "PIANO", "ORQUESTA" o cualquier otro (sintetizador)
 
    SÍNTESIS ADITIVA: un sonido se construye sumando ondas senoidales. Una nota
    con varios armónicos (múltiplos de su frecuencia) suena más rica que un seno
    puro. Después se multiplica por una "envolvente" (curva de volumen en el
    tiempo) para que empiece y acabe suave. Con NumPy se calculan las 44100
    muestras de golpe, sin bucles, así que es instantáneo.
    """
    freqs = np.atleast_1d(np.array(freqs, dtype=float))
    n = int(sr * duracion)        # Nº total de muestras
    i = np.arange(n)              # [0, 1, 2, ... n-1]: índice de cada muestra
    t = i / sr                    # Tiempo en segundos de cada muestra
    # seno(f, k) = onda senoidal de frecuencia k*f (k=2 es el armónico octava)
    seno = lambda f, k=1: np.sin(2 * np.pi * k * f * t)
 
    if estilo == "METRONOMO":
        # Click corto: un seno que se apaga linealmente de 1 a 0
        onda, amp = seno(freqs[0]) * (1 - i / n), 24000
    else:
        amp = 16000   # Amplitud máxima (el límite de 16 bits es 32767)
        if estilo == "PIANO":
            # Ataque rapidísimo (5 ms), caída exponencial y 4 armónicos
            ataque = int(sr * 0.005)
            env = np.exp(-3.5 * t)
            env[:ataque] *= i[:ataque] / ataque
            onda = sum(seno(f) + 0.45 * seno(f, 2) + 0.20 * seno(f, 3) + 0.10 * seno(f, 4) for f in freqs)
        elif estilo == "ORQUESTA":
            # Ataque/relajación lentos, vibrato y voces ligeramente desafinadas
            # (efecto "coro de cuerdas")
            a = r = int(sr * 0.12)
            env = np.clip(np.minimum(i / a, (n - i) / r), 0, 1) * 0.80
            vib = 1.0 + 0.004 * np.sin(2 * np.pi * 5.0 * t)   # Vibrato de 5 Hz
            onda = 0
            for f in freqs:
                fm = f * vib
                onda = onda + (0.50 * np.sin(2 * np.pi * fm * t) + 0.25 * np.sin(2 * np.pi * fm * 1.003 * t) +
                               0.25 * np.sin(2 * np.pi * fm * 0.997 * t) + 0.35 * np.sin(2 * np.pi * 2 * fm * t) +
                               0.15 * np.sin(2 * np.pi * 0.5 * fm * t))
        else:
            # Sintetizador básico: fundamental + 2º armónico con envolvente suave
            # (fade-in y fade-out de 80 ms para evitar "clicks")
            a = r = int(sr * 0.08)
            env = np.clip(np.minimum(i / a, (n - i) / r), 0, 1) * 0.85
            onda = sum(0.65 * seno(f) + 0.35 * seno(f, 2) for f in freqs)
        # Se divide entre el nº de notas para que un acorde no sature, y se aplica la envolvente
        onda = (onda / len(freqs)) * env
 
    # Se pasa a enteros de 16 bits ('<i2' = little-endian) recortando al rango válido
    muestras = np.clip(onda * amp, -32768, 32767).astype('<i2')
    data_size = n * 2   # 2 bytes por muestra
    # Cabecera WAV estándar (44 bytes). Formato struct '<4sI4s4sIHHIIHH4sI':
    #   'RIFF', tamaño total, 'WAVE', 'fmt ', 16, 1=PCM, 1 canal, sample rate,
    #   bytes por segundo, 2 bytes por muestra, 16 bits, 'data', tamaño de datos
    cabecera = struct.pack('<4sI4s4sIHHIIHH4sI', b'RIFF', 36 + data_size, b'WAVE', b'fmt ',
                           16, 1, 1, sr, sr * 2, 2, 16, b'data', data_size)
    return cabecera + muestras.tobytes()
 
 
def cargar_sonido(inst, tipo, clave, archivo, freqs):
    """Registra un sonido: usa el archivo real de /audios si existe; si no, genera uno
    sintético y lo guarda en /audios con el prefijo 'temp_'."""
    ruta = os.path.join(DIR_AUDIOS, archivo)
    if not os.path.exists(ruta):
        ruta = os.path.join(DIR_AUDIOS, "temp_" + archivo)
        with open(ruta, "wb") as f:
            f.write(generar_bytes_wav(freqs, estilo=inst))
    RUTAS_AUDIO[inst][tipo][clave] = ruta
    if PYGAME_DISPONIBLE:
        try:
            # pygame.mixer.Sound carga el .wav en memoria listo para reproducirse sin retardo
            SONIDOS_PYGAME[inst][tipo][clave] = pygame.mixer.Sound(ruta)
        except Exception:
            pass
 
 
def pregenerar_sonidos():
    """Inicializa el mezclador y deja listos el metrónomo y todos los instrumentos."""
    global PYGAME_DISPONIBLE, SONIDO_METRONOMO
    if PYGAME_DISPONIBLE:
        try:
            # pre_init(frecuencia, formato, canales, buffer): buffer pequeño = poca latencia.
            # -16 = enteros de 16 bits con signo; 1 = mono.
            pygame.mixer.pre_init(44100, -16, 1, 512)
            pygame.mixer.init()
            pygame.mixer.set_num_channels(8)           # 8 sonidos simultáneos posibles
            CANALES["loop"] = pygame.mixer.Channel(0)  # Canal reservado al loop grabado
            CANALES["vivo"] = pygame.mixer.Channel(1)  # Canal reservado a lo que tocas
            print("Audio Pygame multicanal inicializado correctamente.")
        except Exception as e:
            print(f"Uso de audio estándar: {e}")
            PYGAME_DISPONIBLE = False
 
    # Click del metrónomo (50 ms a 1000 Hz)
    ruta_metro = os.path.join(DIR_AUDIOS, "temp_metronomo_click.wav")
    with open(ruta_metro, "wb") as f:
        f.write(generar_bytes_wav([1000.0], duracion=0.05, estilo="METRONOMO"))
    if PYGAME_DISPONIBLE:
        try:
            SONIDO_METRONOMO = pygame.mixer.Sound(ruta_metro)
        except Exception:
            SONIDO_METRONOMO = None
 
    # Resto de instrumentos
    for inst, tipo, tabla, nombre_archivo in CATALOGO_AUDIO:
        for clave, freqs in tabla.items():
            cargar_sonido(inst, tipo, clave, nombre_archivo(clave), freqs)
    print("Sintetizador y muestras de instrumentos inicializados correctamente.")
 
 
# ************************************************
# 3. REPRODUCCIÓN
# ***********************************************
def sonar_click_metronomo():
    """Reproduce el click del metrónomo."""
    try:
        if PYGAME_DISPONIBLE and SONIDO_METRONOMO:
            SONIDO_METRONOMO.play()
        elif WINSOUND_DISPONIBLE:
            winsound.Beep(1000, 40)   # Beep(frecuencia Hz, duración ms)
    except Exception:
        pass
 
 
def obtener_recurso_audio(clave, modo_acordes, instrumento):
    """Devuelve el sonido (pygame) o la ruta (winsound) de esa nota/acorde. Si el
    instrumento no lo tiene, usa el sintetizador como reserva."""
    tipo = "ACORDES" if modo_acordes else "NOTAS"
    fuente = SONIDOS_PYGAME if PYGAME_DISPONIBLE else RUTAS_AUDIO
    recurso = fuente.get(instrumento, {}).get(tipo, {}).get(clave)
    return recurso if recurso is not None else fuente["SINTETIZADOR"][tipo].get(clave)
 
 
def reproducir(canal, clave, modo_acordes, instrumento):
    """Reproduce en bucle una nota/acorde en el canal 'vivo' o 'loop'.
    Con la clave SILENCIO corta el sonido de ese canal."""
    if clave == "SILENCIO":
        detener(canal)
        return
    recurso = obtener_recurso_audio(clave, modo_acordes, instrumento)
    if recurso is None:
        return
    if PYGAME_DISPONIBLE and CANALES[canal]:
        CANALES[canal].play(recurso, loops=-1)    # loops=-1: se repite hasta pararlo
    elif WINSOUND_DISPONIBLE and isinstance(recurso, str):
        try:
            winsound.PlaySound(recurso, winsound.SND_FILENAME | winsound.SND_ASYNC | winsound.SND_LOOP)
        except Exception:
            pass
 
 
def detener(canal):
    """Para el sonido de un canal ('vivo' o 'loop'). Con winsound sólo hay un canal."""
    if PYGAME_DISPONIBLE and CANALES[canal]:
        CANALES[canal].stop()
    elif WINSOUND_DISPONIBLE and canal == "vivo":
        try:
            winsound.PlaySound(None, winsound.SND_PURGE)
        except Exception:
            pass
 
 
# ************************************************
# 4. VISIÓN: BRILLO Y DETECCIÓN DE FICHAS
# ************************************************
incremento_brillo = 100   # 100 = sin cambio; <100 oscurece; >100 aclara
 
 
def actualizar_brillo_trackbar(val):
    """Callback del trackbar de brillo: OpenCV la llama cada vez que mueves el deslizador."""
    global incremento_brillo
    incremento_brillo = val
 
 
def ajustar_brillo_frame(img, ajuste):
    """Aclara u oscurece la imagen modificando sólo el canal V (brillo) del HSV."""
    offset = ajuste - 100
    if offset == 0:
        return img
    # cvtColor: BGR -> HSV; split: separa los 3 canales en 3 arrays independientes
    h, s, v = cv2.split(cv2.cvtColor(img, cv2.COLOR_BGR2HSV))
    # Los canales son uint8 (0-255): si sumáramos sin cuidado se desbordaría
    # (250+10 daría 4). Por eso se satura a mano en 255 (o en 0 al restar).
    if offset > 0:
        limite = 255 - offset
        v[v > limite] = 255
        v[v <= limite] += offset
    else:
        k = abs(offset)
        v[v < k] = 0
        v[v >= k] -= k
    # merge: vuelve a unir H, S, V y vuelve a BGR para poder dibujar y mostrar
    return cv2.cvtColor(cv2.merge((h, s, v)), cv2.COLOR_HSV2BGR)
 
 
# Rangos HSV de cada ficha. En OpenCV: H (matiz) va de 0 a 180, S y V de 0 a 255.
# Se ponen S y V mínimos de 50 para descartar grises y zonas oscuras.
# El ROJO usa dos rangos porque su matiz está en los dos extremos del círculo (0 y 180).
FICHAS = [
    ("ROSA",  [(np.array([138, 50, 50]), np.array([168, 255, 255]))], (255, 0, 255)),
    ("AZUL",  [(np.array([95, 50, 50]),  np.array([130, 255, 255]))], (255, 0, 0)),
    ("VERDE", [(np.array([35, 50, 50]),  np.array([85, 255, 255]))],  (0, 255, 0)),
    ("ROJA",  [(np.array([0, 50, 50]),   np.array([8, 255, 255])),
               (np.array([172, 50, 50]), np.array([180, 255, 255]))], (0, 0, 255)),
]
 
 
def evaluar_visibilidad(hsv, rangos):
    """Decide si una ficha es visible. Devuelve (visible, area, contorno_mayor).
 
    Pasos: máscara de color -> limpiar ruido -> buscar manchas -> ¿la mayor es
    suficientemente grande? Si una mano tapa la ficha, la mancha desaparece o
    queda por debajo del umbral y la ficha se considera "tapada".
    """
    mascara = None
    for bajo, alto in rangos:
        # inRange: blanco (255) donde el píxel está entre 'bajo' y 'alto', negro (0) fuera
        m = cv2.inRange(hsv, bajo, alto)
        # bitwise_or: une las máscaras de los distintos rangos (el rojo tiene dos)
        mascara = m if mascara is None else cv2.bitwise_or(mascara, m)
    # Apertura morfológica: erode borra los puntitos sueltos de ruido y dilate
    # devuelve su tamaño a las manchas grandes que sobreviven. None = kernel 3x3 por defecto.
    mascara = cv2.dilate(cv2.erode(mascara, None, iterations=2), None, iterations=2)
    # findContours: lista de contornos de cada mancha blanca.
    # RETR_LIST = todos los contornos sin jerarquía; CHAIN_APPROX_SIMPLE = guarda sólo los vértices.
    contornos, _ = cv2.findContours(mascara, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    if not contornos:
        return False, 0, None
    # Nos quedamos con la mancha más grande (key = su área)
    mayor = max(contornos, key=cv2.contourArea)
    area = cv2.contourArea(mayor)
    return area >= UMBRAL_AREA_MINIMA, area, mayor
 
 
def detectar_fichas(frame):
    """Devuelve una lista (nombre, visible, contorno, color_bgr) con las 4 fichas."""
    # blur: desenfoque 5x5 para suavizar ruido; luego a HSV, donde es fácil filtrar por color
    hsv = cv2.cvtColor(cv2.blur(frame, (5, 5)), cv2.COLOR_BGR2HSV)
    resultado = []
    for nombre, rangos, color in FICHAS:
        visible, _, contorno = evaluar_visibilidad(hsv, rangos)
        resultado.append((nombre, visible, contorno, color))
    return resultado
 
 
# ***************************************************************
# 5. IMÁGENES: BÚSQUEDA, GUARDADO Y TARJETAS
# ***************************************************************
EXTENSIONES = [".png", ".jpg", ".jpeg", ".webp", ".bmp", ".PNG", ".JPG", ".JPEG"]
 
 
def leer_imagen_segura(ruta):
    """Lee una imagen. np.fromfile lee los bytes del archivo y cv2.imdecode los
    convierte en imagen; así funciona también con rutas con tildes en Windows."""
    try:
        return cv2.imdecode(np.fromfile(ruta, dtype=np.uint8), cv2.IMREAD_COLOR)
    except Exception:
        return None
 
 
def guardar_imagen(nombre, img, marca):
    """Guarda una imagen en /tabs como nombre_marca.png y devuelve la ruta."""
    ruta = os.path.join(DIR_TABS, f"{nombre}_{marca}.png")
    # imencode: imagen -> bytes PNG en memoria; tofile: escribe esos bytes en el archivo
    ok, buf = cv2.imencode(".png", img)
    if ok:
        buf.tofile(ruta)
    return ruta
 
 
def buscar_imagen(nombres_base):
    """Busca en /imagenes el primer archivo cuyo nombre (sin extensión) esté en la lista."""
    for base in nombres_base:
        for ext in EXTENSIONES:
            ruta = os.path.join(DIR_IMAGENES, base + ext)
            if os.path.isfile(ruta):
                img = leer_imagen_segura(ruta)
                if img is not None:
                    return img
    return None
 
 
def buscar_imagen_tablatura(tag):
    """Busca la imagen de piano de un acorde (ej. tag 'I_DoM' -> tab_I_DoM, tab_DoM, DoM...)."""
    sub = tag.split("_")[-1]
    bases = [f"tab_{tag}", f"tab_{sub}", tag, sub]
    bases += [b.lower() for b in bases] + [b.upper() for b in bases]
    return buscar_imagen(bases)
 
 
def dibujar_piano_esquematico(ancho=280, alto=130):
    """Teclado de 7 teclas blancas y 5 negras (reserva si no hay imagen de piano)."""
    # np.full crea una imagen (alto x ancho x 3) rellena de 255 = blanco
    img = np.full((alto, ancho, 3), 255, dtype=np.uint8)
    # rectangle(img, esquina_sup_izq, esquina_inf_der, color, grosor)
    cv2.rectangle(img, (0, 0), (ancho - 1, alto - 1), (0, 0, 0), 2)
    w_tb = ancho // 7   # Ancho de cada tecla blanca
    for i in range(7):
        cv2.rectangle(img, (i * w_tb, 0), ((i + 1) * w_tb, alto - 1), (0, 0, 0), 1)
    h_tn, w_tn = int(alto * 0.6), int(w_tb * 0.6)
    for idx in (0, 1, 3, 4, 5):   # Hay tecla negra tras Do, Re, Fa, Sol y La
        x = (idx + 1) * w_tb - w_tn // 2
        cv2.rectangle(img, (x, 0), (x + w_tn, h_tn), (0, 0, 0), -1)   # -1 = relleno
    return img
 
 
def generar_tarjeta_fichas(clave, modo_acordes):
    """Tarjeta 220x160: fichas visibles = círculo de color; tapadas = gris con X roja."""
    visibles = VISIBILIDAD_POR_NOTA[ACORDE_A_NOTA.get(clave, clave)]
    tarjeta = np.zeros((160, 220, 3), dtype=np.uint8)   # zeros = imagen negra
    cv2.rectangle(tarjeta, (0, 0), (220, 160), (255, 255, 255), 2)
    datos = zip(("RS", "A", "V", "RJ"), visibles, (30, 80, 130, 180), [f[2] for f in FICHAS])
    for texto, visible, cx, color in datos:
        cy = 40
        if visible:
            cv2.circle(tarjeta, (cx, cy), 16, color, -1)   # circle(img, centro, radio, color, grosor)
            cv2.putText(tarjeta, texto, (cx - 7, cy + 5), FUENTE, 0.42, (255, 255, 255), 1)
        else:
            cv2.circle(tarjeta, (cx, cy), 16, (60, 60, 60), -1)
            # Dos líneas cruzadas forman la X roja (line(img, punto1, punto2, color, grosor))
            cv2.line(tarjeta, (cx - 10, cy - 10), (cx + 10, cy + 10), (0, 0, 255), 2)
            cv2.line(tarjeta, (cx - 10, cy + 10), (cx + 10, cy - 10), (0, 0, 255), 2)
    cv2.rectangle(tarjeta, (10, 85), (210, 150), (30, 30, 30), -1)
    cv2.rectangle(tarjeta, (10, 85), (210, 150), (180, 180, 180), 1)
    texto = ETIQUETA_CORTA_ACORDE.get(clave, clave) if modo_acordes else f"NOTA: {clave}"
    cv2.putText(tarjeta, texto, (15, 122), FUENTE, 0.55, (255, 255, 255), 1)
    return tarjeta
 
 
def generar_tarjeta_piano(clave, modo_acordes):
    """Tarjeta 300x210 con el dibujo de piano (imagen real o esquemático) y el nombre."""
    tarjeta = np.full((210, 300, 3), 255, dtype=np.uint8)
    cv2.rectangle(tarjeta, (0, 0), (300, 210), (0, 0, 0), 2)
    if modo_acordes:
        tag = MAPEO_CLAVE_ACORDE_ARCHIVO.get(clave, "I_DoM")
        etiqueta = ETIQUETA_CORTA_ACORDE.get(clave, clave)
    else:
        tag = NOTA_A_TAG_ACORDE.get(clave, "I_DoM")
        etiqueta = f"NOTA: {clave}"
 
    img_piano = buscar_imagen_tablatura(tag)
    if img_piano is not None:
        # Escala sin deformar (se usa la menor de las dos proporciones) y centra
        # en el recuadro de 280x130
        h, w = img_piano.shape[:2]
        esc = min(280 / w, 130 / h)
        nw, nh = int(w * esc), int(h * esc)
        ox, oy = 10 + (280 - nw) // 2, 15 + (130 - nh) // 2
        # resize((ancho, alto)) cambia el tamaño; la asignación por "rebanada" pega la
        # imagen en esa zona de la tarjeta
        tarjeta[oy:oy + nh, ox:ox + nw] = cv2.resize(img_piano, (nw, nh))
        cv2.rectangle(tarjeta, (10, 15), (290, 150), (180, 180, 180), 1)
    else:
        tarjeta[15:145, 10:290] = dibujar_piano_esquematico(280, 130)
        cv2.rectangle(tarjeta, (10, 15), (290, 150), (0, 0, 0), 1)
 
    cv2.rectangle(tarjeta, (10, 158), (290, 198), (30, 30, 30), -1)
    cv2.putText(tarjeta, etiqueta, (15, 183), FUENTE, 0.55, (0, 255, 255), 1)
    return tarjeta
 
 
# ***************************************************************
# 6. PENTAGRAMA Y EXPORTACIÓN (a la carpeta /tabs)
# ***************************************************************
# Distancia vertical (px) de cada nota respecto a la 1ª línea del pentagrama (clave de Sol)
OFFSET_NOTAS = {
    "FA5": 0, "MI5": 6, "RE5": 12, "DO5": 18, "SI4": 24, "LA4": 30,
    "SOL4": 36, "FA4": 42, "MI4": 48, "RE4": 54, "DO4": 60
}
# Notas de cada tríada, para dibujarlas apiladas
NOTAS_DE_ACORDE = {
    "I (Do Mayor)": ["DO4", "MI4", "SOL4"], "II (Re menor)": ["RE4", "FA4", "LA4"],
    "III (Mi menor)": ["MI4", "SOL4", "SI4"], "IV (Fa Mayor)": ["FA4", "LA4", "DO5"],
    "V (Sol Mayor)": ["SOL4", "SI4", "RE5"], "VI (La menor)": ["LA4", "DO5", "MI5"],
    "VII (Si dim)": ["SI4", "RE5", "FA5"]
}
 
 
def generar_partitura_pentagrama_ritmica(eventos, marca):
    """Dibuja el pentagrama. La duración sostenida decide la figura:
    >=3s redonda, 1.5-3s blanca, 0.75-1.5s negra, <0.75s corchea."""
    ancho, alto_sis = 1000, 220
    num_sis = math.ceil(len(eventos) / 8)          # 8 notas por pentagrama (ceil = redondeo hacia arriba)
    img = np.full((110 + num_sis * alto_sis, ancho, 3), 255, dtype=np.uint8)
    cv2.rectangle(img, (0, 0), (ancho - 1, 65), (0, 0, 0), 2)
    cv2.putText(img, "PARTITURA MUSICAL EN PENTAGRAMA", (25, 42), FUENTE, 0.75, (0, 0, 0), 2)
 
    paso_x = (ancho - 210) // 8
    for idx_ev, (clave, modo_ac, dur) in enumerate(eventos):
        s, pos = divmod(idx_ev, 8)                 # sistema (fila) y posición dentro de él
        y0 = 140 + s * alto_sis
        if pos == 0:
            # Al empezar cada sistema: 5 líneas, marca de clave y barras inicial/final
            for l in range(5):
                cv2.line(img, (60, y0 + l * 12), (ancho - 40, y0 + l * 12), (0, 0, 0), 1)
            cv2.putText(img, "SOL", (68, y0 + 32), FUENTE, 0.7, (0, 0, 0), 2)
            cv2.line(img, (115, y0), (115, y0 + 48), (0, 0, 0), 2)
            cv2.line(img, (ancho - 40, y0), (ancho - 40, y0 + 48), (0, 0, 0), 2)
 
        x = 150 + pos * paso_x
        if modo_ac:
            notas = NOTAS_DE_ACORDE.get(clave, ["DO4", "MI4", "SOL4"])
            texto = ETIQUETA_CORTA_ACORDE.get(clave, clave)
        else:
            notas, texto = [f"{clave}4"], clave
 
        llena = dur < 1.5                          # negra y corchea: cabeza rellena
        y_alta = 999
        for n in notas:
            off = OFFSET_NOTAS.get(n, 36)
            yn = y0 + off
            y_alta = min(y_alta, yn)
            if off >= 60:                          # Línea adicional para el Do4
                cv2.line(img, (x - 9, y0 + 60), (x + 9, y0 + 60), (0, 0, 0), 1)
            # ellipse(img, centro, (radio_x, radio_y), ángulo, ang_ini, ang_fin, color, grosor):
            # la cabeza de la nota es una elipse inclinada 30 grados
            cv2.ellipse(img, (x, yn), (6, 4), 30, 0, 360, (0, 0, 0), -1 if llena else 2)
 
        if dur < 3.0:                              # Todas menos la redonda llevan plica
            cv2.line(img, (x + 5, y_alta), (x + 5, y_alta - 28), (0, 0, 0), 2)
            if dur < 0.75:                         # Corchete de la corchea
                cv2.line(img, (x + 5, y_alta - 28), (x + 13, y_alta - 18), (0, 0, 0), 2)
        cv2.putText(img, texto, (x - 14, y0 + 72), FUENTE, 0.44, (0, 0, 0), 1)
 
    ruta = guardar_imagen("partitura_ritmica", img, marca)
    mostrar_imagen_segura("Partitura Pentagrama", img)
    print(f"Partitura en pentagrama exportada en: {ruta}")
 
 
def montar_cuadricula(tarjetas, vacia, columnas=4):
    """Une las tarjetas en una cuadrícula, rellenando la última fila con tarjetas vacías."""
    tarjetas = list(tarjetas)
    while len(tarjetas) % columnas:
        tarjetas.append(vacia)
    # np.hstack pega imágenes una al lado de otra (fila); np.vstack una debajo de otra
    return np.vstack([np.hstack(tarjetas[i:i + columnas]) for i in range(0, len(tarjetas), columnas)])
 
 
def exportar_tablaturas(cards_fichas, cards_piano, eventos):
    """Exporta a /tabs tres imágenes: tablatura de fichas, de piano y el pentagrama."""
    if not cards_fichas:
        print("No se registraron notas para exportar tablaturas.")
        return
    marca = time.strftime("%Y%m%d_%H%M%S")   # Fecha y hora: evita sobrescribir exportaciones anteriores
 
    # Tablatura de fichas (fondo negro)
    cuerpo = montar_cuadricula(cards_fichas, np.zeros((160, 220, 3), np.uint8))
    cab = np.zeros((60, cuerpo.shape[1], 3), np.uint8)   # shape[1] = ancho de la imagen
    cv2.putText(cab, "TABLATURA DE GESTOS / FICHAS DE COLOR", (20, 38), FUENTE, 0.7, (0, 255, 255), 2)
    img_f = np.vstack([cab, cuerpo])
    guardar_imagen("tablatura_fichas", img_f, marca)
    mostrar_imagen_segura("Tablatura de Fichas", img_f)
 
    # Tablatura de piano (fondo blanco)
    cuerpo = montar_cuadricula(cards_piano, np.full((210, 300, 3), 255, np.uint8))
    cab = np.full((65, cuerpo.shape[1], 3), 255, np.uint8)
    cv2.rectangle(cab, (0, 0), (cuerpo.shape[1] - 1, 64), (0, 0, 0), 2)
    cv2.putText(cab, "TABLATURA DE DIBUJOS DE PIANO", (20, 42), FUENTE, 0.75, (0, 0, 0), 2)
    img_p = np.vstack([cab, cuerpo])
    guardar_imagen("tablatura_piano", img_p, marca)
    mostrar_imagen_segura("Tablatura de Piano", img_p)
 
    generar_partitura_pentagrama_ritmica(eventos, marca)
    print(f"Exportación terminada en la carpeta: {DIR_TABS}")
 
 
def mostrar_imagen_segura(titulo, img):
    """imshow que no rompe el programa si la ventana no se puede abrir."""
    try:
        cv2.imshow(titulo, img)   # Abre (o refresca) una ventana con ese título
    except Exception:
        pass
 
 
# ************************************************
# 7. HUD AZUL DIVIDIDO: FUNCIONES DE DIBUJO
# ***********************************************
def mezclar_colores(c1, c2, t):
    """Transacciones suaves de color - Interpolación lineal entre dos colores BGR: t=0 -> c1, t=1 -> c2."""
    return tuple(int(a * (1 - t) + b * t) for a, b in zip(c1, c2))
 
 
def puntos_biselados(x0, y0, x1, y1, b):
    """Los 8 vértices de un rectángulo con las esquinas cortadas,
    como los marcos de las láminas. Devuelve un array de enteros para OpenCV."""
    return np.array([(x0 + b, y0), (x1 - b, y0), (x1, y0 + b), (x1, y1 - b),
                     (x1 - b, y1), (x0 + b, y1), (x0, y1 - b), (x0, y0 + b)], dtype=np.int32)
 
 
def panel_translucido(frame, x0, y0, x1, y1, color, alpha, bisel=10):
    """Rellena un panel biselado con un color semitransparente (se sigue viendo
    el vídeo detrás). alpha=0 transparente total, alpha=1 totalmente opaco."""
    capa = frame.copy()                                              # Copia del fotograma
    cv2.fillPoly(capa, [puntos_biselados(x0, y0, x1, y1, bisel)], color)   # Polígono relleno en la copia
    # addWeighted(a, alfa, b, beta, gamma, dst): dst = a*alfa + b*beta + gamma
    # Mezcla la copia pintada con el original => efecto de cristal translúcido
    cv2.addWeighted(capa, alpha, frame, 1 - alpha, 0, dst=frame)
 
 
def marco_biselado(frame, x0, y0, x1, y1, color, grosor=2, bisel=10):
    """Dibuja sólo el contorno de un panel con esquinas cortadas."""
    # polylines(img, [puntos], cerrada, color, grosor, tipo_linea); LINE_AA = bordes suavizados
    cv2.polylines(frame, [puntos_biselados(x0, y0, x1, y1, bisel)], True, color, grosor, cv2.LINE_AA)
 
 
def marco_neon(frame, x0, y0, x1, y1, color, intensidad, bisel=10):
    """Marco con efecto neón: un resplandor ancho y translúcido + una línea nítida encima.
    intensidad (0-1) controla cuánto brilla el resplandor."""
    capa = frame.copy()
    # Línea muy gruesa en la copia = el "halo" de luz alrededor del marco
    cv2.polylines(capa, [puntos_biselados(x0 - 2, y0 - 2, x1 + 2, y1 + 2, bisel)], True, color, 10, cv2.LINE_AA)
    cv2.addWeighted(capa, intensidad, frame, 1 - intensidad, 0, dst=frame)    # Se mezcla con transparencia
    cv2.polylines(frame, [puntos_biselados(x0, y0, x1, y1, bisel)], True, color, 2, cv2.LINE_AA)   # Línea nítida
 
 
def texto(frame, txt, x, y, escala=0.5, color=COLOR_BLANCO, grosor=1):
    """Escribe texto con bordes suavizados. (x, y) = esquina INFERIOR izquierda del texto."""
    # putText(img, texto, origen, fuente, escala, color, grosor, tipo_linea)
    cv2.putText(frame, txt, (x, y), FUENTE, escala, color, grosor, cv2.LINE_AA)
 
 
def texto_centrado(frame, txt, cx, cy, escala, color, grosor=1):
    """Escribe texto centrado en el punto (cx, cy)."""
    # getTextSize devuelve ((ancho, alto), línea_base): sirve para saber cuánto ocupa
    (tw, th), _ = cv2.getTextSize(txt, FUENTE, escala, grosor)
    cv2.putText(frame, txt, (cx - tw // 2, cy + th // 2), FUENTE, escala, color, grosor, cv2.LINE_AA)
 
 
def escala_ajustada(txt, ancho_max, escala_max, grosor):
    """Reduce la escala del texto hasta que quepa en ancho_max píxeles."""
    escala = escala_max
    while escala > 0.4 and cv2.getTextSize(txt, FUENTE, escala, grosor)[0][0] > ancho_max:
        escala -= 0.1
    return escala
 
 
def dibujar_metronomo(frame, x0, y0, x1, y1, met, ahora):
    """Caja del metrónomo (mismo estilo que el HUD): una LUZ circular que late a
    60 BPM, sin números, y debajo el tempo y el estado del sonido (tecla T).
    En cada pulso la luz se enciende en amarillo y se agranda; durante el resto
    del segundo se apaga y se encoge poco a poco."""
    cx = (x0 + x1) // 2                      # Centro horizontal de la caja
    cy = y0 + 44                             # Centro vertical de la luz
 
    # Fondo translúcido y marco cian, igual que el HUD principal
    panel_translucido(frame, x0, y0, x1, y1, COLOR_FONDO, 0.80, bisel=8)
    marco_biselado(frame, x0, y0, x1, y1, COLOR_CIAN, 2, bisel=8)
 
    # 'fase' va de 0 (justo en el pulso) a 1 (a punto del siguiente pulso)
    fase = min(1.0, (ahora - met.t_ultimo) / met.intervalo)
    radio = int(12 + 16 * (1 - fase) ** 2)   # De 28 px (pulso) a 12 px (reposo)
    luz = mezclar_colores(COLOR_AMARILLO, COLOR_FONDO, min(1.0, fase * 1.2))   # Se apaga hacia el fondo
    aro = mezclar_colores(COLOR_AMARILLO, COLOR_CIAN, fase)
 
    # Halo translúcido alrededor de la luz (más fuerte justo en el pulso)
    capa = frame.copy()
    cv2.circle(capa, (cx, cy), radio + 8, aro, -1, cv2.LINE_AA)
    alfa = 0.08 + 0.30 * (1 - fase)
    cv2.addWeighted(capa, alfa, frame, 1 - alfa, 0, dst=frame)
 
    cv2.circle(frame, (cx, cy), radio, luz, -1, cv2.LINE_AA)   # Luz (relleno, grosor -1)
    cv2.circle(frame, (cx, cy), radio, aro, 2, cv2.LINE_AA)    # Aro de color
 
    # Texto: tempo y estado del click. Después de los dos puntos va minúscula.
    texto_centrado(frame, "60 BPM", cx, cy + 48, 0.55, COLOR_BLANCO, 1)
    estado = "on" if met.sonido else "off"
    txt = f"T: click {estado}"
    escala = escala_ajustada(txt, (x1 - x0) - 12, 0.5, 1)      # Se encoge si no cabe en la caja
    texto_centrado(frame, txt, cx, cy + 70, escala, COLOR_CIAN if met.sonido else (205, 185, 140), 1)
 
 
def dibujar_hud(frame, it, met, loop, partitura, ahora):
    """HUD azul dividido en secciones.
 
        +-------------------------------+-----------------+   (  )
        | INSTRUMENTO: Piano            |                 |   (60 BPM)
        | MODO: Acordes                 |       DoM       |
        +-------------------------------+--------+--------+
        | Loopstation: inactivo         | pulsa R para    |
        | Partitura: pulsa P para ...   | grabar          |
        +-------------------------------+-----------------+
    """
    h, w = frame.shape[:2]                  # shape = (alto, ancho, canales)
    x0, y0 = 15, 12                         # Esquina superior izquierda del panel
    x1 = w - 125                            # Se deja sitio a la derecha para el círculo
    y_sep = y0 + 80                         # Línea que separa la zona superior de la inferior
    y1 = y_sep + 58                         # Borde inferior del panel
    x_sep = x0 + int((x1 - x0) * 0.55)      # Línea vertical que separa texto y nota
 
    # 1) Fondo azul marino translúcido y marco cian biselado
    panel_translucido(frame, x0, y0, x1, y1, COLOR_FONDO, 0.80)
    marco_biselado(frame, x0, y0, x1, y1, COLOR_CIAN, 2)
 
    # 2) Divisiones internas (líneas finas)
    cv2.line(frame, (x0, y_sep), (x1, y_sep), COLOR_CIAN, 1, cv2.LINE_AA)
    cv2.line(frame, (x_sep, y0), (x_sep, y_sep + 28), COLOR_CIAN, 1, cv2.LINE_AA)
    cv2.line(frame, (x0 + 10, y_sep + 28), (x1 - 10, y_sep + 28), COLOR_APAGADO, 1, cv2.LINE_AA)
 
    # 3) Sección superior izquierda: instrumento y modo
    texto(frame, f"INSTRUMENTO: {it.instrumento.lower()}", x0 + 14, y0 + 34, 0.55, COLOR_BLANCO, 1)
    texto(frame, f"MODO: {'acordes' if it.modo_acordes else 'notas'}", x0 + 14, y0 + 64, 0.55, COLOR_CIAN, 1)
 
    # 4) Sección superior derecha: la nota/acorde activo, grande y en amarillo brillante
    if it.clave == "SILENCIO":
        nota, color_nota = "--", COLOR_APAGADO
    else:
        nota = ETIQUETA_CORTA_ACORDE.get(it.clave, it.clave)    # DoM, Rem... o DO, RE...
        color_nota = COLOR_AMARILLO
    escala = escala_ajustada(nota, (x1 - x_sep) - 24, 2.2, 4)   # Se encoge si no cabe
    texto_centrado(frame, nota, (x_sep + x1) // 2, y0 + 40, escala, color_nota, 4)
 
    # 5) Sección inferior: estado del loop (izquierda) y qué hacer (derecha)
    linea_estado, linea_accion, color_loop = loop.texto_estado(ahora)
    texto(frame, linea_estado, x0 + 14, y_sep + 20, 0.5, COLOR_BLANCO, 1)
    texto(frame, linea_accion, x_sep + 12, y_sep + 20, 0.45, color_loop, 1)
 
    # 6) Fila inferior completa: estado de la exportación de partitura
    if partitura.activo:
        txt_part = f"Partitura: grabando {len(partitura.eventos)} notas - pulsa P para exportar"
        color_part = COLOR_ROJO
    else:
        txt_part = "Partitura: pulsa P para exportar partitura y tablaturas"
        color_part = COLOR_CIAN
    texto(frame, txt_part, x0 + 14, y_sep + 47, 0.42, color_part, 1)
 
    # 7) Feedback visual de respuesta: mientras suena una nota, el marco del HUD se
    #    ilumina en amarillo neón. Al empezar la nota brilla más y luego baja a un
    #    brillo sostenido (decaimiento exponencial con math.exp).
    if it.clave != "SILENCIO":
        edad = ahora - it.t_nota                           # Segundos desde que empezó la nota
        intensidad = 0.25 + 0.30 * math.exp(-edad * 6)
        marco_neon(frame, x0, y0, x1, y1, COLOR_AMARILLO, intensidad)
 
    # 8) Caja del metrónomo a la derecha del HUD (luz que late + texto)
    dibujar_metronomo(frame, x1 + 10, y0, w - 15, y1, met, ahora)
 
 
# *************************************************
# 8. TUTORIAL SUPERPUESTO (misma ventana, con botón X)
# *************************************************
# Texto de reserva por si no existen las imágenes lamina_N en /imagenes.
# (cv2.putText no dibuja tildes ni ñ, por eso van sin acentos.)
TUTORIAL_TEXTOS = [
    ("1. INTRODUCCION", ["Sintetizador gestual por vision artificial.",
                         "Tapa o destapa las 4 fichas de colores",
                         "(rosa, azul, verde y roja) para tocar."]),
    ("2. COMBINACIONES", ["Todas visibles: DO / DoM (I)", "Rosa tapada: RE / Rem (II)",
                          "Azul tapada: MI / Mim (III)", "Verde tapada: FA / FaM (IV)",
                          "Roja tapada: SOL / SolM (V)", "Rosa + Azul tapadas: LA / Lam (VI)",
                          "Rosa + Verde tapadas: SI / Sidim (VII)"]),
    ("3. MODOS E INSTRUMENTOS", ["M: alterna Acordes / Notas individuales", "I: cambia de instrumento",
                                 "Acordes: Sintetizador, Piano, Orquesta",
                                 "Notas: Sintetizador, Piano, Violonchelo"]),
    ("4. LOOPSTATION", ["R: graba en el siguiente pulso (60 BPM)",
                        "R otra vez: cierra y cuantiza el bucle",
                        "El loop suena mientras tocas encima", "C: borra el loop"]),
    ("5. EXPORTACION Y CONTROLES", ["P: exporta partitura y 2 tablaturas (/tabs)",
                                    "T: sonido del metronomo   H: tutorial", "Q / ESC: salir"]),
]
 
# Códigos que devuelve waitKeyEx para las flechas (cambian según el sistema operativo)
TECLAS_IZQUIERDA = {2424832, 65361, 2}   # Flechas izquierda (Windows / Linux)
TECLAS_DERECHA = {2555904, 65363, 3}     # Flechas derecha
 
 
class Tutorial:
    """Tutorial que se dibuja encima del vídeo, más pequeño que la ventana.
    Se cierra con ESC / ENTER / H o con clic en la [X]."""
    TOTAL = 5
    ALTO_BARRA = 30   # Altura de la barra inferior con las instrucciones
 
    def __init__(self):
        self.visible = True          # Empieza abierto al arrancar
        self.indice = 0
        self.rects = {}              # Zonas clicables: nombre -> (x, y, ancho, alto)
        self.originales = {}         # Láminas cargadas del disco (o None si no existen)
        self.cache = {}              # Láminas ya redimensionadas (para no repetir el trabajo)
 
    # ---- Estado ----
    def abrir(self):
        self.visible = True
 
    def cerrar(self):
        self.visible = False
 
    def mover(self, paso):
        # El operador % hace que al pasar de la última lámina vuelva a la primera
        self.indice = (self.indice + paso) % self.TOTAL
 
    # ---- Entrada ----
    def manejar_tecla(self, k):
        """Devuelve True si el tutorial está abierto (en ese caso consume la tecla).
        Nota: A/D sólo sirven para navegar por las láminas del tutorial."""
        if not self.visible:
            return False
        base = k & 0xFF    # Se queda con el byte bajo: el código ASCII de la tecla
        if base in (27, 13, ord('h'), ord('H')):                      # ESC, ENTER, H
            self.cerrar()
        elif k in TECLAS_IZQUIERDA or base in (ord('a'), ord('A')):
            self.mover(-1)
        elif k in TECLAS_DERECHA or base in (32, ord('d'), ord('D')):  # espacio, D
            self.mover(1)
        return True
 
    def al_hacer_clic(self, evento, x, y, flags, param):
        """Callback del ratón (registrada con cv2.setMouseCallback). OpenCV la llama con
        cada evento: 'evento' es el tipo (clic, movimiento...) y (x, y) la posición."""
        if not self.visible or evento != cv2.EVENT_LBUTTONDOWN:   # Sólo clic izquierdo
            return
        acciones = {"cerrar": self.cerrar, "anterior": lambda: self.mover(-1), "siguiente": lambda: self.mover(1)}
        for nombre, (rx, ry, rw, rh) in self.rects.items():
            if rx <= x <= rx + rw and ry <= y <= ry + rh:   # ¿El clic cae dentro del botón?
                acciones[nombre]()
                return
 
    # ---- Dibujo ----
    def _lamina_texto(self, ancho, alto):
        """Lámina de reserva dibujada con texto, escalada al tamaño pedido."""
        esc = ancho / 800.0
        img = np.full((alto, ancho, 3), 20, dtype=np.uint8)
        cv2.rectangle(img, (3, 3), (ancho - 4, alto - 4), COLOR_CIAN, 1)
        titulo, lineas = TUTORIAL_TEXTOS[self.indice]
        cv2.putText(img, titulo, (int(30 * esc), int(65 * esc)), FUENTE, max(0.5, 0.85 * esc), COLOR_CIAN, 1)
        for i, linea in enumerate(lineas):
            cv2.putText(img, "- " + linea, (int(30 * esc), int((125 + i * 48) * esc)),
                        FUENTE, max(0.42, 0.7 * esc), (255, 255, 255), 1)
        return img
 
    def _obtener_lamina(self, ancho, alto):
        """Devuelve la lámina actual del tamaño pedido (imagen real si existe)."""
        clave = (self.indice, ancho, alto)
        if clave not in self.cache:
            if self.indice not in self.originales:
                n = f"lamina_{self.indice + 1}"
                self.originales[self.indice] = buscar_imagen([n, n.capitalize(), n.upper()])
            orig = self.originales[self.indice]
            # INTER_AREA es la interpolación recomendada al REDUCIR una imagen (más nítida)
            self.cache[clave] = (cv2.resize(orig, (ancho, alto), interpolation=cv2.INTER_AREA)
                                 if orig is not None else self._lamina_texto(ancho, alto))
        return self.cache[clave]
 
    def dibujar(self, frame):
        """Dibuja el tutorial centrado sobre el frame de la cámara."""
        if not self.visible:
            return
        h, w = frame.shape[:2]
        ancho = int(w * 0.82)                       # Ocupa ~82% del ancho
        alto = int(ancho / 1.5)
        if alto > h * 0.78:                          # Si no cabe en alto, se ajusta
            alto = int(h * 0.78)
            ancho = int(alto * 1.5)
        x0 = (w - ancho) // 2
        y0 = max(0, (h - alto - self.ALTO_BARRA) // 2)
 
        # Pegar la lámina: frame[y0:y1, x0:x1] = imagen sustituye esa zona de píxeles
        frame[y0:y0 + alto, x0:x0 + ancho] = self._obtener_lamina(ancho, alto)
        yb = y0 + alto
        cv2.rectangle(frame, (x0, yb), (x0 + ancho, yb + self.ALTO_BARRA), (30, 30, 30), -1)
        cv2.rectangle(frame, (x0, y0), (x0 + ancho, yb + self.ALTO_BARRA), COLOR_CIAN, 1)
        txt = f"ESC / [X]: cerrar  |  < > : navegar  |  {self.indice + 1}/{self.TOTAL}"
        tam = cv2.getTextSize(txt, FUENTE, 0.4, 1)[0]    # Para centrar el texto en la barra
        cv2.putText(frame, txt, (x0 + (ancho - tam[0]) // 2, yb + 20), FUENTE, 0.4, (255, 255, 255), 1)
 
        # Botones: [X] arriba a la derecha, < y > en la barra. Se guardan sus rectángulos
        # para que al_hacer_clic sepa dónde están.
        b = 26
        self.rects = {
            "cerrar": (x0 + ancho - b - 6, y0 + 6, b, b),
            "anterior": (x0 + 6, yb + 3, 24, 24),
            "siguiente": (x0 + ancho - 30, yb + 3, 24, 24),
        }
        rx, ry, _, _ = self.rects["cerrar"]
        cv2.rectangle(frame, (rx, ry), (rx + b, ry + b), (0, 0, 200), -1)          # Fondo rojo
        cv2.rectangle(frame, (rx, ry), (rx + b, ry + b), (255, 255, 255), 1)       # Borde blanco
        cv2.line(frame, (rx + 7, ry + 7), (rx + b - 7, ry + b - 7), (255, 255, 255), 2)   # Diagonal 1 de la X
        cv2.line(frame, (rx + b - 7, ry + 7), (rx + 7, ry + b - 7), (255, 255, 255), 2)   # Diagonal 2 de la X
        for nombre, simbolo in (("anterior", "<"), ("siguiente", ">")):
            bx, by, bw, bh = self.rects[nombre]
            cv2.rectangle(frame, (bx, by), (bx + bw, by + bh), (255, 255, 255), 1)
            cv2.putText(frame, simbolo, (bx + 7, by + 17), FUENTE, 0.55, (255, 255, 255), 1)
 
 
# ***************************************************************
# 9. COMPONENTES DE LA APLICACIÓN
# ***************************************************************
class Metronomo:
    """Reloj de 60 BPM (1 pulso por segundo).
 
    El RELOJ siempre corre (así el círculo del HUD se enciende y la loopstation puede
    sincronizarse), pero el SONIDO del click sólo se oye si el usuario lo activa con T.
    Por defecto el sonido empieza DESACTIVADO.
    """
 
    def __init__(self, bpm=60):
        self.sonido = False                  # ¿Suena el click? (tecla T). Empieza apagado.
        self.intervalo = 60.0 / bpm          # Segundos entre pulsos (60 BPM = 1,0 s)
        self.t_ultimo = time.time()          # Instante del último pulso
        self.contador = 1                    # Número de pulsos desde el inicio
 
    def actualizar(self, ahora):
        """Devuelve True si en este instante toca un nuevo pulso."""
        if ahora - self.t_ultimo >= self.intervalo:
            self.t_ultimo = ahora
            self.contador += 1
            return True
        return False
 
    def alternar_sonido(self):
        """Tecla T: activa/desactiva el sonido del click."""
        self.sonido = not self.sonido
 
 
class Interprete:
    """Convierte la combinación de fichas en una nota/acorde estable y gestiona
    el modo (acordes/notas) y el instrumento."""
 
    def __init__(self):
        self.modo_acordes = True
        self.idx_inst = 0
        self.reiniciar()
 
    def reiniciar(self):
        """Vuelve al silencio (se usa al cambiar de modo)."""
        self.clave = "SILENCIO"      # Nota confirmada que suena ahora
        self.candidata = "SILENCIO"  # Nota detectada que aún debe estabilizarse
        self.t_candidata = time.time()
        self.anterior = "SILENCIO"   # Última nota confirmada (para medir su duración)
        self.t_nota = time.time()    # Instante en que empezó la nota actual
 
    @property
    def instrumento(self):
        return INSTRUMENTOS[self.modo_acordes][self.idx_inst]
 
    def cambiar_instrumento(self):
        self.idx_inst = (self.idx_inst + 1) % len(INSTRUMENTOS[self.modo_acordes])
 
    def cambiar_modo(self):
        self.modo_acordes = not self.modo_acordes
        self.idx_inst = 0
        self.reiniciar()
 
    def clave_de(self, visibles):
        """Combinación de fichas visibles -> nota o acorde (SILENCIO si no es válida)."""
        nota = NOTA_POR_PATRON.get(visibles)
        if nota is None:
            return "SILENCIO"
        return NOTA_A_ACORDE[nota] if self.modo_acordes else nota
 
    def actualizar(self, visibles, ahora):
        """Aplica el filtro anti-rebote. Si se confirma una nota nueva devuelve
        (nota_anterior, duracion_sostenida); si no, devuelve None."""
        instante = self.clave_de(visibles)
        if instante != self.candidata:
            # El gesto ha cambiado: se reinicia el cronómetro de estabilidad
            self.candidata, self.t_candidata = instante, ahora
            return None
        if ahora - self.t_candidata < TIEMPO_MINIMO_ESTABILIDAD or self.candidata == self.clave:
            return None
        self.clave = self.candidata
        if self.clave == self.anterior:
            return None
        cerrada = (self.anterior, ahora - self.t_nota)
        self.anterior, self.t_nota = self.clave, ahora
        return cerrada
 
 
class Loopstation:
    """Estados: IDLE -> ARMED_REC (espera pulso) -> RECORDING -> PLAYING."""
 
    def __init__(self):
        self.estado = 'IDLE'
        self.eventos = []      # (tiempo_relativo, clave, modo_acordes, instrumento)
        self.duracion = 0.0
        self.t_inicio = 0.0    # Inicio de la grabación
        self.t_repro = 0.0     # Inicio de la reproducción (alineado a un pulso)
        self.indice = -1
 
    def _evento(self, t, interprete):
        return (t, interprete.clave, interprete.modo_acordes, interprete.instrumento)
 
    def al_pulso(self, ahora, interprete):
        """Si está esperando el pulso, empieza a grabar EXACTAMENTE en este pulso."""
        if self.estado == 'ARMED_REC':
            self.estado, self.t_inicio = 'RECORDING', ahora
            self.eventos = [self._evento(0.0, interprete)]   # Lo que ya sonaba en ese instante
            print("Pulso marcado: grabación de loop iniciada en sincronía.")
 
    def registrar(self, ahora, interprete):
        """Anota un cambio de nota/instrumento mientras se graba."""
        if self.estado == 'RECORDING':
            self.eventos.append(self._evento(ahora - self.t_inicio, interprete))
 
    def pulsar_r(self, ahora, metronomo):
        """Tecla R: empieza, cancela, cierra+cuantiza o reinicia según el estado."""
        if self.estado == 'IDLE':
            self.estado = 'ARMED_REC'
            print("Loopstation lista: grabará en el próximo pulso.")
        elif self.estado == 'ARMED_REC':
            self.estado = 'IDLE'
            print("Grabación cancelada.")
        elif self.estado == 'RECORDING':
            # Cuantización: se redondea la duración al número entero de pulsos más cercano
            beats = max(1, round((ahora - self.t_inicio) / metronomo.intervalo))
            self.duracion = beats * metronomo.intervalo
            self.eventos = [e for e in self.eventos if e[0] < self.duracion]
            if self.eventos:
                self.estado, self.t_repro, self.indice = 'PLAYING', metronomo.t_ultimo, -1
                print(f"Loop cuantizado a {beats} pulso(s) ({self.duracion:.1f}s). Reproduciendo...")
            else:
                self.estado = 'IDLE'
        else:  # PLAYING: R vuelve a preparar otra grabación
            self.estado = 'ARMED_REC'
            detener("loop")
            print("Nueva grabación de loop (esperando pulso)...")
 
    def borrar(self):
        """Tecla C: borra por completo el loop."""
        self.estado, self.eventos, self.duracion = 'IDLE', [], 0.0
        detener("loop")
        print("Loop borrado.")
 
    def reproducir(self, ahora):
        """Reproduce el bucle: cambia de sonido al pasar de un evento al siguiente."""
        if self.estado != 'PLAYING' or self.duracion <= 0 or not self.eventos:
            return
        t = (ahora - self.t_repro) % self.duracion     # Posición dentro del bucle (vuelve a 0 al final)
        idx = max((i for i, e in enumerate(self.eventos) if t >= e[0]), default=0)
        if idx != self.indice:
            _, clave, modo, inst = self.eventos[idx]
            reproducir("loop", clave, modo, inst)
            self.indice = idx
 
    def texto_estado(self, ahora):
        """Devuelve (linea_estado, linea_accion, color_bgr) para el HUD."""
        if self.estado == 'IDLE':
            return "Loopstation: inactivo", "pulsa R para grabar", COLOR_CIAN
        if self.estado == 'ARMED_REC':
            return "Loopstation: activo", "esperando pulso...", COLOR_CIAN
        if self.estado == 'RECORDING':
            return "Loopstation: activo", f"grabando {ahora - self.t_inicio:.1f}s - R: cerrar", COLOR_ROJO
        return "Loopstation: activo", f"sonando {self.duracion:.0f}s - C: borrar", COLOR_VERDE
 
 
class GrabadorPartitura:
    """Registra las notas tocadas entre dos pulsaciones de P y las exporta a /tabs."""
 
    def __init__(self):
        self.activo = False
        self.eventos, self.cards_fichas, self.cards_piano = [], [], []
 
    def cerrar_nota(self, clave, modo_acordes, duracion):
        """Añade una nota terminada (si dura lo suficiente) a las tres salidas a la vez."""
        if self.activo and clave != "SILENCIO" and duracion >= UMBRAL_MINIMO_RECORD_NOTA:
            self.eventos.append((clave, modo_acordes, duracion))
            self.cards_fichas.append(generar_tarjeta_fichas(clave, modo_acordes))
            self.cards_piano.append(generar_tarjeta_piano(clave, modo_acordes))
 
    def alternar(self, interprete, ahora):
        """Primera pulsación: empieza a grabar. Segunda: cierra la última nota y exporta."""
        if not self.activo:
            self.eventos.clear(); self.cards_fichas.clear(); self.cards_piano.clear()
            self.activo = True
            interprete.t_nota = ahora   # La nota en curso cuenta desde ahora
            print("Iniciada grabación de partitura y tablaturas...")
        else:
            self.cerrar_nota(interprete.clave, interprete.modo_acordes, ahora - interprete.t_nota)
            self.activo = False
            exportar_tablaturas(self.cards_fichas, self.cards_piano, self.eventos)
 
 
class SintetizadorGestual:
    """Une todos los componentes. main() sólo tiene que llamar a procesar(), dibujar() y tecla()."""
 
    def __init__(self):
        self.metronomo = Metronomo()
        self.interprete = Interprete()
        self.loop = Loopstation()
        self.partitura = GrabadorPartitura()
        self.tutorial = Tutorial()
        self.fichas = []
 
    def procesar(self, frame):
        """Metrónomo, detección de fichas, sonido en vivo y loop. Devuelve el frame ajustado."""
        ahora = time.time()
        if self.metronomo.actualizar(ahora):
            if self.metronomo.sonido:          # El click sólo suena si se activó con T
                sonar_click_metronomo()
            self.loop.al_pulso(ahora, self.interprete)   # La loopstation siempre usa el pulso
 
        frame = ajustar_brillo_frame(frame, incremento_brillo)
        self.fichas = detectar_fichas(frame)
        # Con el tutorial abierto no se toca (se fuerza el silencio)
        visibles = None if self.tutorial.visible else tuple(f[1] for f in self.fichas)
 
        cambio = self.interprete.actualizar(visibles, ahora)
        if cambio:
            anterior, duracion = cambio
            self.partitura.cerrar_nota(anterior, self.interprete.modo_acordes, duracion)
            reproducir("vivo", self.interprete.clave, self.interprete.modo_acordes, self.interprete.instrumento)
            self.loop.registrar(ahora, self.interprete)
 
        self.loop.reproducir(ahora)
        return frame
 
    def dibujar(self, frame):
        """Dibuja las fichas detectadas, el HUD y, encima de todo, el tutorial."""
        ahora = time.time()
 
        # Recuadro y nombre alrededor de cada ficha visible
        for nombre, visible, contorno, color in self.fichas:
            if visible and contorno is not None:
                # boundingRect: rectángulo mínimo (x, y, ancho, alto) que envuelve el contorno
                x, y, w, h = cv2.boundingRect(contorno)
                cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
                cv2.putText(frame, nombre, (x, y - 10), FUENTE, 0.6, color, 2)
 
        dibujar_hud(frame, self.interprete, self.metronomo, self.loop, self.partitura, ahora)
        self.tutorial.dibujar(frame)   # Se dibuja el último para quedar por encima de todo
 
    def tecla(self, k):
        """Procesa una tecla (valor de waitKeyEx). Devuelve False para salir."""
        if self.tutorial.manejar_tecla(k):   # Con el tutorial abierto, él se queda la tecla
            return True
        # k & 0xFF = código ASCII; chr() lo convierte en carácter; lower() ignora mayúsculas
        c = chr(k & 0xFF).lower() if k != -1 else ''
        ahora = time.time()
        it = self.interprete
 
        if c == 't':
            self.metronomo.alternar_sonido()
            if self.metronomo.sonido:
                sonar_click_metronomo()      # Click inmediato para confirmar que está activo
            print(f"Sonido del metrónomo: {'ACTIVADO' if self.metronomo.sonido else 'DESACTIVADO'}")
        elif c == 'i':
            it.cambiar_instrumento()
            reproducir("vivo", it.clave, it.modo_acordes, it.instrumento)
            self.loop.registrar(ahora, it)   # Si se graba un loop, el cambio de timbre queda anotado
            print(f"Instrumento: {it.instrumento}")
        elif c == 'm':
            # Antes de cambiar de modo se cierra la nota en curso de la partitura
            self.partitura.cerrar_nota(it.anterior, it.modo_acordes, ahora - it.t_nota)
            it.cambiar_modo()
            detener("vivo")   # Las claves de un modo no existen en el otro
            print(f"Modo: {'ACORDES' if it.modo_acordes else 'NOTAS'} | Instrumento: {it.instrumento}")
        elif c == 'p':
            self.partitura.alternar(it, ahora)
        elif c == 'r':
            self.loop.pulsar_r(ahora, self.metronomo)
        elif c == 'c':
            self.loop.borrar()
        elif c == 'h':
            self.tutorial.abrir()
        elif c in ('q', chr(27)):
            return False
        return True
 
    def cerrar(self):
        """Para todo el audio al salir."""
        detener("vivo")
        detener("loop")
 
 
# ************************************************
# 10. PROGRAMA PRINCIPAL
# ************************************************
def abrir_camara():
    """Abre la cámara 1; si no existe, prueba la 0. Devuelve None si no hay ninguna."""
    for indice in (1, 0):
        camara = cv2.VideoCapture(indice)   # Abre la cámara número 'indice'
        if camara.isOpened():               # ¿Se abrió correctamente?
            return camara
        camara.release()                    # Libera el recurso antes de probar la siguiente
    return None
 
 
def main():
    crear_carpetas()
    if WINSOUND_DISPONIBLE or PYGAME_DISPONIBLE:
        pregenerar_sonidos()
 
    camara = abrir_camara()
    if camara is None:
        print("Error: No es posible abrir la cámara web.")
        return
 
    ventana = "Gesture Synth OpenCV + Loopstation"
    app = SintetizadorGestual()
    cv2.namedWindow(ventana)                                                 # Crea la ventana
    cv2.createTrackbar("Ajuste Brillo", ventana, 100, 200, actualizar_brillo_trackbar)  # Deslizador 0-200
    cv2.setMouseCallback(ventana, app.tutorial.al_hacer_clic)   # Para el botón [X] del tutorial
 
    while True:
        ret, frame = camara.read()                 # Captura un fotograma (ret = ¿tuvo éxito?)
        if not ret:
            break
        frame = app.procesar(cv2.flip(frame, 1))   # flip(...,1) = espejo horizontal, más natural
        app.dibujar(frame)
        mostrar_imagen_segura(ventana, frame)
        if not app.tecla(cv2.waitKeyEx(1)):        # Espera 1 ms una tecla y refresca la ventana
            break
 
    app.cerrar()
    camara.release()          # Libera la cámara
    cv2.destroyAllWindows()   # Cierra las ventanas de OpenCV
 
 
if __name__ == '__main__':
    main()
 