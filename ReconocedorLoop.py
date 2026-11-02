import cv2
import numpy as np
import struct
import math
import time
import os

# Carga de libreria de audio multicanal
try:
    import pygame
    PYGAME_DISPONIBLE = True
except ImportError:
    PYGAME_DISPONIBLE = False

try:
    import winsound
    WINSOUND_DISPONIBLE = True
except ImportError:
    WINSOUND_DISPONIBLE = False

# ==========================================
# 1. DEFINICION DE FRECUENCIAS Y MAPEOS
# ==========================================
NOTAS_SOLO = {
    "DO":  [261.63],
    "RE":  [293.66],
    "MI":  [329.63],
    "FA":  [349.23],
    "SOL": [392.00],
    "LA":  [440.00],
    "SI":  [493.88]
}

ACORDES = {
    "I (Do Mayor)":     [261.63, 329.63, 392.00],   # C - E - G
    "II (Re menor)":    [293.66, 349.23, 440.00],   # D - F - A
    "III (Mi menor)":   [329.63, 392.00, 493.88],   # E - G - B
    "IV (Fa Mayor)":    [349.23, 440.00, 523.25],   # F - A - C5
    "V (Sol Mayor)":    [392.00, 493.88, 587.33],   # G - B - D5
    "VI (La menor)":    [440.00, 523.25, 659.25],   # A - C5 - E5
    "VII (Si dism)":    [493.88, 587.33, 698.46]    # B - D5 - F5
}

MAPEO_CLAVE_ACORDE_ARCHIVO = {
    "I (Do Mayor)":     "I_DoM",
    "II (Re menor)":    "II_Rem",
    "III (Mi menor)":   "III_Mim",
    "IV (Fa Mayor)":    "IV_FaM",
    "V (Sol Mayor)":    "V_SolM",
    "VI (La menor)":    "VI_Lam",
    "VII (Si dism)":    "VII_Sidism"
}

# Diccionarios de rutas WAV
RUTAS_AUDIO = {
    "SINTETIZADOR": {"NOTAS": {}, "ACORDES": {}},
    "PIANO":        {"NOTAS": {}, "ACORDES": {}},
    "VIOLONCHELO":  {"NOTAS": {}, "ACORDES": {}},
    "ORQUESTA":     {"NOTAS": {}, "ACORDES": {}}
}

# Diccionarios de Sonidos Pygame
SONIDOS_PYGAME = {
    "SINTETIZADOR": {"NOTAS": {}, "ACORDES": {}},
    "PIANO":        {"NOTAS": {}, "ACORDES": {}},
    "VIOLONCHELO":  {"NOTAS": {}, "ACORDES": {}},
    "ORQUESTA":     {"NOTAS": {}, "ACORDES": {}}
}

canal_loop = None
canal_vivo = None

incremento_brillo = 100  # 100 es neutro (rango 0 a 200)

def actualizar_brillo_trackbar(val):
    global incremento_brillo
    incremento_brillo = val

def ajustar_brillo_frame(img, ajuste):
    """Ajusta la luminosidad modificando el canal V (Value) en espacio HSV."""
    offset = ajuste - 100
    if offset == 0:
        return img
    
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)
    
    if offset > 0:
        limite = 255 - offset
        v[v > limite] = 255
        v[v <= limite] += offset
    else:
        abs_offset = abs(offset)
        v[v < abs_offset] = 0
        v[v >= abs_offset] -= abs_offset
        
    hsv = cv2.merge((h, s, v))
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

# ==========================================
# 2. SINTESIS Y CARGA DE AUDIOS (1.0 seg)
# ==========================================
def generar_bytes_wav(lista_frecuencias, duracion=1.0, sample_rate=44100):
    num_samples = int(sample_rate * duracion)
    data_size = num_samples * 2
    header = struct.pack(
        '<4sI4s4sIHHIIHH4sI',
        b'RIFF', 36 + data_size, b'WAVE', b'fmt ',
        16, 1, 1, sample_rate, sample_rate * 2, 2, 16,
        b'data', data_size
    )
    samples = bytearray()
    attack = int(sample_rate * 0.08)
    release = int(sample_rate * 0.08)
    num_notas = len(lista_frecuencias)
    
    for i in range(num_samples):
        t = i / sample_rate
        mezcla_onda = 0.0
        for freq in lista_frecuencias:
            mezcla_onda += (0.65 * math.sin(2 * math.pi * freq * t) + 
                            0.35 * math.sin(2 * math.pi * 2 * freq * t))
        mezcla_onda = mezcla_onda / num_notas
        
        envolvente = 0.85
        if i < attack:
            envolvente = (i / attack) * 0.85
        elif i > num_samples - release:
            envolvente = ((num_samples - i) / release) * 0.85
            
        val = int(mezcla_onda * envolvente * 16000)
        val = max(-32768, min(32767, val))
        samples.extend(struct.pack('<h', val))
        
    return header + samples

def registrar_audio(inst, tipo, clave, ruta):
    if os.path.exists(ruta):
        RUTAS_AUDIO[inst][tipo][clave] = ruta
        if PYGAME_DISPONIBLE:
            try:
                SONIDOS_PYGAME[inst][tipo][clave] = pygame.mixer.Sound(ruta)
            except Exception:
                pass
    else:
        ruta_synth = RUTAS_AUDIO["SINTETIZADOR"][tipo].get(clave)
        if ruta_synth and os.path.exists(ruta_synth):
            RUTAS_AUDIO[inst][tipo][clave] = ruta_synth
            if PYGAME_DISPONIBLE:
                SONIDOS_PYGAME[inst][tipo][clave] = SONIDOS_PYGAME["SINTETIZADOR"][tipo].get(clave)

def pregenerar_sonidos():
    global PYGAME_DISPONIBLE, canal_loop, canal_vivo
    
    if PYGAME_DISPONIBLE:
        try:
            pygame.mixer.pre_init(44100, -16, 1, 512)
            pygame.mixer.init()
            pygame.mixer.set_num_channels(8)
            canal_loop = pygame.mixer.Channel(0)
            canal_vivo = pygame.mixer.Channel(1)
            print("Audio Pygame multicanal inicializado correctamente.")
        except Exception as e:
            print(f"Uso de audio estandar: {e}")
            PYGAME_DISPONIBLE = False

    # 1. Sintetizador (Generacion matematica a 1.0s)
    for nombre, freqs in NOTAS_SOLO.items():
        data = generar_bytes_wav(freqs, duracion=1.0)
        ruta = f"temp_synth_nota_{nombre}.wav"
        with open(ruta, "wb") as f:
            f.write(data)
        RUTAS_AUDIO["SINTETIZADOR"]["NOTAS"][nombre] = ruta
        if PYGAME_DISPONIBLE:
            SONIDOS_PYGAME["SINTETIZADOR"]["NOTAS"][nombre] = pygame.mixer.Sound(ruta)
        
    for nombre, freqs in ACORDES.items():
        clave_limpia = "".join(e for e in nombre if e.isalnum())
        data = generar_bytes_wav(freqs, duracion=1.0)
        ruta = f"temp_synth_acorde_{clave_limpia}.wav"
        with open(ruta, "wb") as f:
            f.write(data)
        RUTAS_AUDIO["SINTETIZADOR"]["ACORDES"][nombre] = ruta
        if PYGAME_DISPONIBLE:
            SONIDOS_PYGAME["SINTETIZADOR"]["ACORDES"][nombre] = pygame.mixer.Sound(ruta)

    # 2. Cello (Notas individuales)
    for nombre in NOTAS_SOLO.keys():
        registrar_audio("VIOLONCHELO", "NOTAS", nombre, f"cello_{nombre}.wav")

    # 3. Piano (Notas y Acordes)
    for nombre in NOTAS_SOLO.keys():
        registrar_audio("PIANO", "NOTAS", nombre, f"piano_nota_{nombre}.wav")
        
    for nombre in ACORDES.keys():
        tag = MAPEO_CLAVE_ACORDE_ARCHIVO[nombre]
        registrar_audio("PIANO", "ACORDES", nombre, f"piano_acorde_{tag}.wav")

    # 4. Orquesta (Acordes)
    for nombre in ACORDES.keys():
        tag = MAPEO_CLAVE_ACORDE_ARCHIVO[nombre]
        registrar_audio("ORQUESTA", "ACORDES", nombre, f"orquesta_acorde_{tag}.wav")

    print("Sintetizador y muestras de instrumentos inicializados correctamente.")

# ==========================================
# 3. REPRODUCCION MULTICANAL
# ==========================================
def obtener_recurso_audio(clave_actual, modo_acordes, instrumento):
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
    if clave_actual == "SILENCIO":
        detener_audio_vivo()
        return
        
    recurso = obtener_recurso_audio(clave_actual, modo_acordes, instrumento)
    if PYGAME_DISPONIBLE and canal_vivo and recurso:
        canal_vivo.play(recurso, loops=-1)
    elif WINSOUND_DISPONIBLE and recurso and isinstance(recurso, str):
        try:
            winsound.PlaySound(recurso, winsound.SND_FILENAME | winsound.SND_ASYNC | winsound.SND_LOOP)
        except Exception:
            pass

def reproducir_sonido_loop(clave, modo_acordes, instrumento):
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
    if PYGAME_DISPONIBLE and canal_vivo:
        canal_vivo.stop()
    elif WINSOUND_DISPONIBLE:
        try:
            winsound.PlaySound(None, winsound.SND_PURGE)
        except Exception:
            pass

def detener_audio_loop():
    if PYGAME_DISPONIBLE and canal_loop:
        canal_loop.stop()

# ==========================================
# 4. RANGOS DE COLOR HSV Y VISIBILIDAD
# ==========================================
rosa_bajo = np.array([140, 80, 80])
rosa_alto = np.array([175, 255, 255])

azul_bajo = np.array([90, 80, 80])
azul_alto = np.array([130, 255, 255])

verde_bajo = np.array([35, 80, 80])
verde_alto = np.array([85, 255, 255])

amarillo_bajo = np.array([20, 80, 80])
amarillo_alto = np.array([34, 255, 255])

UMBRAL_AREA_MINIMA = 3000

def evaluar_visibilidad(hsv_frame, bajo, alto):
    mascara = cv2.inRange(hsv_frame, bajo, alto)
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
# 5. GENERACION DE PARTITURA VISUAL / TABLATURA
# ==========================================
def generar_tarjeta_nota(clave, rosa_vis, azul_vis, verde_vis, amarillo_vis):
    """
    Crea una miniatura (200x160 px) que representa visualmente la posicion de la mano 
    y el nombre del acorde/nota grabado.
    """
    tarjeta = np.zeros((160, 200, 3), dtype=np.uint8)
    cv2.rectangle(tarjeta, (0, 0), (200, 160), (255, 255, 255), 2)
    
    fichas_coords = [
        ("R", rosa_vis, (25, 40), (255, 0, 255)),
        ("A", azul_vis, (75, 40), (255, 0, 0)),
        ("V", verde_vis, (125, 40), (0, 255, 0)),
        ("Y", amarillo_vis, (175, 40), (0, 255, 255))
    ]
    
    for nombre_f, visible, (cx, cy), color_bgr in fichas_coords:
        if visible:
            cv2.circle(tarjeta, (cx, cy), 14, color_bgr, -1)
            cv2.putText(tarjeta, nombre_f[0], (cx - 5, cy + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 1)
        else:
            cv2.circle(tarjeta, (cx, cy), 14, (60, 60, 60), -1)
            cv2.line(tarjeta, (cx - 8, cy - 8), (cx + 8, cy + 8), (0, 0, 255), 2)
            cv2.line(tarjeta, (cx - 8, cy + 8), (cx + 8, cy - 8), (0, 0, 255), 2)
            
    cv2.rectangle(tarjeta, (10, 85), (190, 145), (30, 30, 30), -1)
    cv2.rectangle(tarjeta, (10, 85), (190, 145), (180, 180, 180), 1)
    cv2.putText(tarjeta, clave, (15, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    return tarjeta

def exportar_partitura_final(lista_tarjetas, nombre_salida="partitura_generada.png"):
    """
    Organiza las tarjetas de notas en filas de 4 columnas formando una lamina de partitura visual.
    """
    if len(lista_tarjetas) == 0:
        print("No se grabaron eventos para exportar la partitura.")
        return
        
    columnas = 4
    filas = math.ceil(len(lista_tarjetas) / columnas)
    
    tarjetas_completas = list(lista_tarjetas)
    tarjeta_vacía = np.zeros((160, 200, 3), dtype=np.uint8)
    while len(tarjetas_completas) % columnas != 0:
        tarjetas_completas.append(tarjeta_vacía)
        
    filas_concatenadas = []
    for f in range(filas):
        bloque_fila = tarjetas_completas[f * columnas : (f + 1) * columnas]
        fila_img = np.hstack(bloque_fila)
        filas_concatenadas.append(fila_img)
        
    partitura_final = np.vstack(filas_concatenadas)
    
    # Encabezado
    ancho_p = partitura_final.shape[1]
    encabezado = np.zeros((60, ancho_p, 3), dtype=np.uint8)
    cv2.putText(encabezado, "PARTITURA VISUAL / TABLATURA DE GESTOS", (20, 38), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
    
    partitura_completa = np.vstack([encabezado, partitura_final])
    
    cv2.imwrite(nombre_salida, partitura_completa)
    cv2.imshow("Partitura Visual Generada", partitura_completa)
    print(f"Partitura visual exportada con exito en: {nombre_salida}")

# ==========================================
# 6. BUCLE PRINCIPAL (MAIN)
# ==========================================
def main():
    if WINSOUND_DISPONIBLE or PYGAME_DISPONIBLE:
        pregenerar_sonidos()
        
    camara = cv2.VideoCapture(1)
    
    if not camara.isOpened():
        print("Error: No es posible abrir la camara web.")
        return

    nombre_ventana = "Gesture Synth OpenCV + Loopstation"
    cv2.namedWindow(nombre_ventana)
    cv2.createTrackbar("Ajuste Brillo", nombre_ventana, 100, 200, actualizar_brillo_trackbar)

    insts_acordes = ["SINTETIZADOR", "PIANO", "ORQUESTA"]
    insts_notas = ["SINTETIZADOR", "PIANO", "VIOLONCHELO"]
    
    modo_acordes = True
    instrumentos_actuales = insts_acordes
    idx_instrumento = 0
    instrumento_actual = instrumentos_actuales[idx_instrumento]

    clave_actual = "SILENCIO"
    clave_anterior = None

    # Loopstation
    estado_loop = 'IDLE'
    tiempo_inicio_grabacion = 0
    duracion_loop = 0
    eventos_grabados = []
    tiempo_inicio_reproduccion = 0
    indice_evento_loop = -1

    # Partitura Visual
    grabando_partitura = False
    lista_tarjetas_partitura = []

    while True:
        ret, frame = camara.read()
        if not ret:
            break
            
        # Volteo horizontal para efecto espejo natural
        frame = cv2.flip(frame, 1)
        t_actual = time.time()
        
        # 1. Ajuste de brillo
        frame_ajustado = ajustar_brillo_frame(frame, incremento_brillo)
        
        frame_suavizado = cv2.blur(frame_ajustado, (5, 5))
        hsv = cv2.cvtColor(frame_suavizado, cv2.COLOR_BGR2HSV)
        
        # 2. Visibilidad HSV
        rosa_visible, _, c_rosa = evaluar_visibilidad(hsv, rosa_bajo, rosa_alto)
        azul_visible, _, c_azul = evaluar_visibilidad(hsv, azul_bajo, azul_alto)
        verde_visible, _, c_verde = evaluar_visibilidad(hsv, verde_bajo, verde_alto)
        amarillo_visible, _, c_amarillo = evaluar_visibilidad(hsv, amarillo_bajo, amarillo_alto)
        
        rosa_tapado = not rosa_visible
        azul_tapado = not azul_visible
        verde_tapado = not verde_visible
        amarillo_tapado = not amarillo_visible
        
        # 3. MAPEO ESTRICTO CON COMPROBACION BIDIRECCIONAL Y CONTROL DE ERRORES
        if rosa_visible and azul_visible and verde_visible and amarillo_visible:
            clave_actual = "I (Do Mayor)" if modo_acordes else "DO"
        elif rosa_tapado and azul_tapado and verde_visible and amarillo_visible:
            clave_actual = "VI (La menor)" if modo_acordes else "LA"
        elif rosa_tapado and verde_tapado and azul_visible and amarillo_visible:
            clave_actual = "VII (Si dism)" if modo_acordes else "SI"
        elif rosa_tapado and azul_visible and verde_visible and amarillo_visible:
            clave_actual = "II (Re menor)" if modo_acordes else "RE"
        elif azul_tapado and rosa_visible and verde_visible and amarillo_visible:
            clave_actual = "III (Mi menor)" if modo_acordes else "MI"
        elif verde_tapado and rosa_visible and azul_visible and amarillo_visible:
            clave_actual = "IV (Fa Mayor)" if modo_acordes else "FA"
        elif amarillo_tapado and rosa_visible and azul_visible and verde_visible:
            clave_actual = "V (Sol Mayor)" if modo_acordes else "SOL"
        else:
            # Estado no valido / Oscuridad / Mas de 2 fichas tapadas -> SILENCIO
            clave_actual = "SILENCIO"

        # 4. Reproduccion en vivo y grabacion de eventos
        if clave_actual != clave_anterior:
            reproducir_sonido_vivo(clave_actual, modo_acordes, instrumento_actual)
            
            if estado_loop == 'RECORDING':
                t_rel = t_actual - tiempo_inicio_grabacion
                eventos_grabados.append((t_rel, clave_actual, modo_acordes, instrumento_actual))
                
            if grabando_partitura and clave_actual != "SILENCIO":
                tarjeta = generar_tarjeta_nota(clave_actual, rosa_visible, azul_visible, verde_visible, amarillo_visible)
                lista_tarjetas_partitura.append(tarjeta)
                
            clave_anterior = clave_actual

        # 5. Reproduccion de Loopstation de fondo
        if estado_loop == 'PLAYING' and duracion_loop > 0 and len(eventos_grabados) > 0:
            t_en_loop = (t_actual - tiempo_inicio_reproduccion) % duracion_loop
            
            idx_evento = 0
            for i, (t_ev, _, _, _) in enumerate(eventos_grabados):
                if t_en_loop >= t_ev:
                    idx_evento = i
                else:
                    break
            
            if idx_evento != indice_evento_loop:
                _, clave_loop, modo_loop_ev, inst_loop_ev = eventos_grabados[idx_evento]
                reproducir_sonido_loop(clave_loop, modo_loop_ev, inst_loop_ev)
                indice_evento_loop = idx_evento

        # 6. Dibujado de HUD
        fichas = [
            ("ROSA", rosa_visible, c_rosa, (255, 0, 255)),
            ("AZUL", azul_visible, c_azul, (255, 0, 0)),
            ("VERDE", verde_visible, c_verde, (0, 255, 0)),
            ("AMARILLA", amarillo_visible, c_amarillo, (0, 255, 255))
        ]
        
        for nombre, visible, contorno, color_bgr in fichas:
            if visible and contorno is not None:
                x, y, w, h = cv2.boundingRect(contorno)
                cv2.rectangle(frame_ajustado, (x, y), (x + w, y + h), color_bgr, 2)
                cv2.putText(frame_ajustado, nombre, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color_bgr, 2)

        # Panel HUD Principal
        cv2.rectangle(frame_ajustado, (20, 15), (610, 150), (0, 0, 0), -1)
        cv2.rectangle(frame_ajustado, (20, 15), (610, 150), (255, 255, 255), 2)
        
        modo_str = "ACORDES" if modo_acordes else "NOTAS INDIVIDUALES"
        
        cv2.putText(frame_ajustado, f"INSTRUMENTO: {instrumento_actual} | MODO: {modo_str}", (30, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 1)
        cv2.putText(frame_ajustado, f"EN VIVO: {clave_actual}", (30, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        
        if estado_loop == 'IDLE':
            loop_str = "LOOPSTATION: Inactivo (Pulsa 'R' para grabar)"
            color_loop = (200, 200, 200)
        elif estado_loop == 'RECORDING':
            tiempo_rec = t_actual - tiempo_inicio_grabacion
            loop_str = f"GRABANDO LOOP... ({tiempo_rec:.1f}s) - Pulsa 'R' para terminar"
            color_loop = (0, 0, 255)
        elif estado_loop == 'PLAYING':
            loop_str = f"REPRODUCIENDO LOOP ({duracion_loop:.1f}s) - Pulsa 'C' para borrar"
            color_loop = (0, 255, 0)
            
        cv2.putText(frame_ajustado, loop_str, (30, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color_loop, 2)
        
        part_str = f"PARTITURA: GRABANDO ({len(lista_tarjetas_partitura)} notas)" if grabando_partitura else "PARTITURA: Pulsa 'P' para generar tablatura"
        color_part = (0, 0, 255) if grabando_partitura else (180, 180, 180)
        cv2.putText(frame_ajustado, part_str, (30, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.42, color_part, 1)

        cv2.putText(frame_ajustado, f"['I': Inst | 'M': Modo | 'R': Loop | 'P': Partitura | 'C': Borrar | Brillo: {incremento_brillo}]", (30, 140), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (180, 180, 180), 1)

        cv2.imshow(nombre_ventana, frame_ajustado)
        
        # 7. Controles de Teclado
        key = cv2.waitKey(1) & 0xFF
        
        # 'I': Cambiar Instrumento dentro del modo activo
        if key in [ord('i'), ord('I')]:
            idx_instrumento = (idx_instrumento + 1) % len(instrumentos_actuales)
            instrumento_actual = instrumentos_actuales[idx_instrumento]
            clave_anterior = None
            print(f"Instrumento cambiado a: {instrumento_actual}")

        # 'M': Cambiar Modo (Acordes / Notas)
        elif key in [ord('m'), ord('M')]:
            modo_acordes = not modo_acordes
            instrumentos_actuales = insts_acordes if modo_acordes else insts_notas
            idx_instrumento = 0
            instrumento_actual = instrumentos_actuales[idx_instrumento]
            clave_anterior = None
            print(f"Modo cambiado a: {'ACORDES' if modo_acordes else 'NOTAS'} | Instrumento: {instrumento_actual}")

        # 'P': Modo Partitura Visual
        elif key in [ord('p'), ord('P')]:
            if not grabando_partitura:
                grabando_partitura = True
                lista_tarjetas_partitura = []
                print("Iniciada grabacion de partitura visual...")
            else:
                grabando_partitura = False
                print("Exportando partitura visual...")
                exportar_partitura_final(lista_tarjetas_partitura)

        # 'R': Loopstation
        elif key in [ord('r'), ord('R')]:
            if estado_loop == 'IDLE':
                estado_loop = 'RECORDING'
                tiempo_inicio_grabacion = t_actual
                eventos_grabados = [(0.0, clave_actual, modo_acordes, instrumento_actual)]
                print("Grabacion de Loop iniciada...")
            elif estado_loop == 'RECORDING':
                duracion_loop = max(0.5, t_actual - tiempo_inicio_grabacion)
                if len(eventos_grabados) > 0:
                    estado_loop = 'PLAYING'
                    tiempo_inicio_reproduccion = t_actual
                    indice_evento_loop = -1
                    print(f"Loop grabado ({duracion_loop:.2f} s). Reproduciendo...")
                else:
                    estado_loop = 'IDLE'
            elif estado_loop == 'PLAYING':
                estado_loop = 'RECORDING'
                detener_audio_loop()
                tiempo_inicio_grabacion = t_actual
                eventos_grabados = [(0.0, clave_actual, modo_acordes, instrumento_actual)]
                print("Reiniciando grabacion de nuevo Loop...")

        # 'C': Limpiar Loopstation
        elif key in [ord('c'), ord('C')]:
            estado_loop = 'IDLE'
            eventos_grabados = []
            duracion_loop = 0
            detener_audio_loop()
            print("Loop borrado.")

        elif key in [ord('q'), ord('Q'), 27]:
            break

    detener_audio_vivo()
    detener_audio_loop()
    if WINSOUND_DISPONIBLE:
        try:
            winsound.PlaySound(None, winsound.SND_PURGE)
        except Exception:
            pass
        
    camara.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()