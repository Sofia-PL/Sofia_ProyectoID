import cv2
import numpy as np
import struct
import math
import time

# Carga de librería de audio multicanal
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
# 1. NOTAS Y ACORDES (SINTETIZADOR POLIFÓNICO)
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
    "V (Sol Mayor7)":   [392.00, 493.88, 587.33, 698.46], # G - B - D5 - F5
    "VI (La menor)":    [440.00, 523.25, 659.25],   # A - C5 - E5
    "VII (Si dism)":    [493.88, 587.33, 698.46]    # B - D5 - F5
}

RUTAS_NOTAS_WAV = {}
RUTAS_ACORDES_WAV = {}
SONIDOS_PYGAME_NOTAS = {}
SONIDOS_PYGAME_ACORDES = {}

canal_loop = None
canal_vivo = None

# Variable global para el ajuste interactivo de brillo mediante Trackbar
incremento_brillo = 100  # 100 es el valor neutro inicial (rango 0 a 200)

def actualizar_brillo_trackbar(val):
    global incremento_brillo
    incremento_brillo = val

def ajustar_brillo_frame(img, ajuste):
    """
    Ajusta la luminosidad del fotograma modificando el canal V (Value) en HSV.
    Técnica oficial de la asignatura (Ejercicio 28 / VIDEO BARRA).
    """
    offset = ajuste - 100  # Convertir rango  a offset [-100, +100]
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

def generar_bytes_wav(lista_frecuencias, duracion=1.5, sample_rate=44100):
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

def pregenerar_sonidos():
    global RUTAS_NOTAS_WAV, RUTAS_ACORDES_WAV, PYGAME_DISPONIBLE, canal_loop, canal_vivo
    
    if PYGAME_DISPONIBLE:
        try:
            pygame.mixer.pre_init(44100, -16, 1, 512)
            pygame.mixer.init()
            pygame.mixer.set_num_channels(8)
            canal_loop = pygame.mixer.Channel(0)
            canal_vivo = pygame.mixer.Channel(1)
            print("Audio Pygame inicializado con soporte multicanal (Loop + Vivo).")
        except Exception as e:
            print(f"Uso de audio estándar: {e}")
            PYGAME_DISPONIBLE = False

    for nombre, freqs in NOTAS_SOLO.items():
        data = generar_bytes_wav(freqs)
        ruta = f"temp_nota_{nombre}.wav"
        with open(ruta, "wb") as f:
            f.write(data)
        RUTAS_NOTAS_WAV[nombre] = ruta
        if PYGAME_DISPONIBLE:
            SONIDOS_PYGAME_NOTAS[nombre] = pygame.mixer.Sound(ruta)
        
    for nombre, freqs in ACORDES.items():
        clave_limpia = "".join(e for e in nombre if e.isalnum())
        data = generar_bytes_wav(freqs)
        ruta = f"temp_acorde_{clave_limpia}.wav"
        with open(ruta, "wb") as f:
            f.write(data)
        RUTAS_ACORDES_WAV[nombre] = ruta
        if PYGAME_DISPONIBLE:
            SONIDOS_PYGAME_ACORDES[nombre] = pygame.mixer.Sound(ruta)
        
    print("Sintetizador inicializado correctamente.")

def reproducir_sonido_vivo(clave_actual, modo_acordes):
    if PYGAME_DISPONIBLE and canal_vivo:
        sound_dict = SONIDOS_PYGAME_ACORDES if modo_acordes else SONIDOS_PYGAME_NOTAS
        if clave_actual in sound_dict:
            canal_vivo.play(sound_dict[clave_actual], loops=-1)
    elif WINSOUND_DISPONIBLE:
        ruta_dict = RUTAS_ACORDES_WAV if modo_acordes else RUTAS_NOTAS_WAV
        ruta = ruta_dict.get(clave_actual)
        if ruta:
            try:
                winsound.PlaySound(ruta, winsound.SND_FILENAME | winsound.SND_ASYNC | winsound.SND_LOOP)
            except Exception:
                pass

def reproducir_sonido_loop(clave, modo_acordes):
    if PYGAME_DISPONIBLE and canal_loop:
        sound_dict = SONIDOS_PYGAME_ACORDES if modo_acordes else SONIDOS_PYGAME_NOTAS
        if clave in sound_dict:
            canal_loop.play(sound_dict[clave], loops=-1)
    elif WINSOUND_DISPONIBLE:
        ruta_dict = RUTAS_ACORDES_WAV if modo_acordes else RUTAS_NOTAS_WAV
        ruta = ruta_dict.get(clave)
        if ruta:
            try:
                winsound.PlaySound(ruta, winsound.SND_FILENAME | winsound.SND_ASYNC | winsound.SND_LOOP)
            except Exception:
                pass

def detener_audio_vivo():
    if PYGAME_DISPONIBLE and canal_vivo:
        canal_vivo.stop()

def detener_audio_loop():
    if PYGAME_DISPONIBLE and canal_loop:
        canal_loop.stop()

# ==========================================
# 2. RANGOS DE COLOR EN ESPACIO HSV
# ==========================================
# ROSA FOSFORITO (Subrayador)
rosa_bajo = np.array([140, 100, 100])
rosa_alto = np.array([170, 255, 255])

# AZUL
azul_bajo = np.array([95, 100, 100])
azul_alto = np.array([135, 255, 255])

# VERDE (Subrayador verde)
verde_bajo = np.array([40, 80, 80])
verde_alto = np.array([85, 255, 255])

# AMARILLO (Subrayador amarillo)
amarillo_bajo = np.array([20, 100, 100])
amarillo_alto = np.array([35, 255, 255])

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

def main():
    if WINSOUND_DISPONIBLE or PYGAME_DISPONIBLE:
        pregenerar_sonidos()
        
    camara = cv2.VideoCapture(1)
    
    if not camara.isOpened():
        print("Error: No es posible abrir la cámara web.")
        return

    # Crear ventana interactiva con Trackbar de Brillo (Ejercicio VIDEO BARRA)
    nombre_ventana = "Gesture Synth OpenCV + Loopstation"
    cv2.namedWindow(nombre_ventana)
    cv2.createTrackbar("Ajuste Brillo", nombre_ventana, 100, 200, actualizar_brillo_trackbar)

    print("=== SINTETIZADOR DE GESTOS + LOOPSTATION ACTIVO ===")
    print("Controles:")
    print(" - Mueve el deslizador 'Ajuste Brillo' en la ventana para compensar la luz ambiental.")
    print(" - Pulsa 'R' para Empezar a Grabar / Detener y Repetir Bucle (Loop)")
    print(" - Pulsa 'C' para Borrar el Bucle grabado")
    print(" - Pulsa 'M' para cambiar entre MODO NOTAS y MODO ACORDES")
    print(" - Pulsa 'q' o 'ESC' para salir.")

    modo_acordes = True
    clave_actual = ""
    clave_anterior = None

    # Variables de la Loopstation
    estado_loop = 'IDLE'  # 'IDLE', 'RECORDING', 'PLAYING'
    tiempo_inicio_grabacion = 0
    duracion_loop = 0
    eventos_grabados = []
    
    tiempo_inicio_reproduccion = 0
    indice_evento_loop = -1

    while True:
        ret, frame = camara.read()
        if not ret:
            break
            
        t_actual = time.time()
        
        # 1. Ajuste dinámico de brillo en tiempo real antes de binarizar
        frame_ajustado = ajustar_brillo_frame(frame, incremento_brillo)
        
        frame_suavizado = cv2.blur(frame_ajustado, (5, 5))
        hsv = cv2.cvtColor(frame_suavizado, cv2.COLOR_BGR2HSV)
        
        # 2. Visibilidad y segmentación sobre la imagen corregida
        rosa_visible, _, c_rosa = evaluar_visibilidad(hsv, rosa_bajo, rosa_alto)
        azul_visible, _, c_azul = evaluar_visibilidad(hsv, azul_bajo, azul_alto)
        verde_visible, _, c_verde = evaluar_visibilidad(hsv, verde_bajo, verde_alto)
        amarillo_visible, _, c_amarillo = evaluar_visibilidad(hsv, amarillo_bajo, amarillo_alto)
        
        rosa_tapado = not rosa_visible
        azul_tapado = not azul_visible
        verde_tapado = not verde_visible
        amarillo_tapado = not amarillo_visible
        
        # Combinaciones musicales
        if rosa_tapado and azul_tapado:
            clave_actual = "VI (La menor)" if modo_acordes else "LA"
            estado_texto = "Rosa + Azul Tapadas"
        elif rosa_tapado and verde_tapado:
            clave_actual = "VII (Si dism)" if modo_acordes else "SI"
            estado_texto = "Rosa + Verde Tapadas"
        elif rosa_tapado:
            clave_actual = "II (Re menor)" if modo_acordes else "RE"
            estado_texto = "Rosa Tapada"
        elif azul_tapado:
            clave_actual = "III (Mi menor)" if modo_acordes else "MI"
            estado_texto = "Azul Tapada"
        elif verde_tapado:
            clave_actual = "IV (Fa Mayor)" if modo_acordes else "FA"
            estado_texto = "Verde Tapada"
        elif amarillo_tapado:
            clave_actual = "V (Sol Mayor7)" if modo_acordes else "SOL"
            estado_texto = "Amarilla Tapada"
        else:
            clave_actual = "I (Do Mayor)" if modo_acordes else "DO"
            estado_texto = "Todas visibles"

        # 3. Reproducción en vivo
        if clave_actual != clave_anterior:
            reproducir_sonido_vivo(clave_actual, modo_acordes)
            
            if estado_loop == 'RECORDING':
                t_rel = t_actual - tiempo_inicio_grabacion
                eventos_grabados.append((t_rel, clave_actual, modo_acordes))
                
            clave_anterior = clave_actual

        # 4. Reproducción del Loop grabado de fondo
        if estado_loop == 'PLAYING' and duracion_loop > 0 and len(eventos_grabados) > 0:
            t_en_loop = (t_actual - tiempo_inicio_reproduccion) % duracion_loop
            
            idx_evento = 0
            for i, (t_ev, _, _) in enumerate(eventos_grabados):
                if t_en_loop >= t_ev:
                    idx_evento = i
                else:
                    break
            
            if idx_evento != indice_evento_loop:
                _, clave_loop, modo_loop_ev = eventos_grabados[idx_evento]
                reproducir_sonido_loop(clave_loop, modo_loop_ev)
                indice_evento_loop = idx_evento

        # 5. Interfaz Gráfica (HUD)
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

        # Panel HUD de control
        cv2.rectangle(frame_ajustado, (20, 20), (560, 140), (0, 0, 0), -1)
        cv2.rectangle(frame_ajustado, (20, 20), (560, 140), (255, 255, 255), 2)
        
        modo_str = "MODO: ACORDES (Gesture-Synth)" if modo_acordes else "MODO: NOTAS INDIVIDUALES"
        cv2.putText(frame_ajustado, modo_str, (30, 45), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
        cv2.putText(frame_ajustado, f"EN VIVO: {clave_actual}", (30, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        
        if estado_loop == 'IDLE':
            loop_str = "LOOPSTATION: Inactivo (Pulsa 'R' para grabar)"
            color_loop = (200, 200, 200)
        elif estado_loop == 'RECORDING':
            tiempo_rec = t_actual - tiempo_inicio_grabacion
            loop_str = f" GRABANDO LOOP... ({tiempo_rec:.1f}s) - Pulsa 'R' para terminar"
            color_loop = (0, 0, 255)
        elif estado_loop == 'PLAYING':
            loop_str = f"REPRODUCIENDO LOOP ({duracion_loop:.1f}s) - Pulsa 'C' para borrar"
            color_loop = (0, 255, 0)
            
        cv2.putText(frame_ajustado, loop_str, (30, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color_loop, 2)
        cv2.putText(frame_ajustado, f"['R': Rec/Stop | 'C': Borrar | 'M': Modo | Nivel Brillo: {incremento_brillo}]", (30, 125), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (180, 180, 180), 1)

        cv2.imshow(nombre_ventana, frame_ajustado)
        
        # Teclas de control
        key = cv2.waitKey(1) & 0xFF
        
        if key in [ord('r'), ord('R')]:
            if estado_loop == 'IDLE':
                estado_loop = 'RECORDING'
                tiempo_inicio_grabacion = t_actual
                eventos_grabados = [(0.0, clave_actual, modo_acordes)]
                print("Grabación de Loop iniciada...")
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
                eventos_grabados = [(0.0, clave_actual, modo_acordes)]
                print("Reiniciando grabación de nuevo Loop...")

        elif key in [ord('c'), ord('C')]:
            estado_loop = 'IDLE'
            eventos_grabados = []
            duracion_loop = 0
            detener_audio_loop()
            print("Loop borrado.")

        elif key in [ord('m'), ord('M')]:
            modo_acordes = not modo_acordes
            clave_anterior = None

        elif key in [ord('q'), ord('Q'), 27]:
            break

    detener_audio_vivo()
    detener_audio_loop()
    if WINSOUND_DISPONIBLE:
        winsound.PlaySound(None, winsound.SND_PURGE)
        
    camara.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()