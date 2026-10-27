import cv2
import numpy as np
import struct
import math

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
    "I (Do Mayor)":   [261.63, 329.63, 392.00],   # C - E - G
    "II (Re menor)":  [293.66, 349.23, 440.00],   # D - F - A
    "III (Mi menor)": [329.63, 392.00, 493.88],   # E - G - B
    "IV (Fa Mayor)":  [349.23, 440.00, 523.25],   # F - A - C5
    "V (Sol Mayor)":  [392.00, 493.88, 587.33, 698.46],   # G - B - D5 - F5
    "VI (La menor)":  [440.00, 523.25, 659.25],   # A - C5 - E5
    "VII (Si dism)":  [493.88, 587.33, 698.46]    # B - D5 - F5
}

RUTAS_NOTAS_WAV = {}
RUTAS_ACORDES_WAV = {}
# sample_rate = 44100  # Frecuencia de muestreo estándar para audio
# num_samples = int(sample_rate * duracion) numero de muestras para la duración deseada
#data_size = num_samples * 2  # 2 bytes por muestra (16 bits)
def generar_wav_acorde(lista_frecuencias, duracion=1.5, sample_rate=44100):
    num_samples = int(sample_rate * duracion)
    data_size = num_samples * 2

    # Crear el encabezado WAV para que cualquier reproductor de audio reconoczca el .wav
    header = struct.pack(
        '<4sI4s4sIHHIIHH4sI',
        b'RIFF',
        36 + data_size,
        b'WAVE',
        b'fmt ',
        16, 1, 1, sample_rate, sample_rate * 2, 2, 16,
        b'data',
        data_size
    )

    # Modificamos envolvente para que no haya clics al inicio y al final del sonido
    samples = bytearray()
    attack = int(sample_rate * 0.08)   # 80 ms de ataque suave estilo pad
    release = int(sample_rate * 0.08)  # 80 ms de desvanecimiento suave
    num_notas = len(lista_frecuencias)

    # Generar la mezcla de ondas para el acorde
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

# Preparar audio .wav para cada nota y acorde, para que se reproduzcan de manera instantánea
def pregenerar_sonidos():
    global RUTAS_NOTAS_WAV, RUTAS_ACORDES_WAV
    
    for nombre, freqs in NOTAS_SOLO.items():
        data = generar_wav_acorde(freqs)
        ruta = f"temp_nota_{nombre}.wav"
        with open(ruta, "wb") as f:
            f.write(data)
        RUTAS_NOTAS_WAV[nombre] = ruta
        
    for nombre, freqs in ACORDES.items():
        data = generar_wav_acorde(freqs)
        ruta = f"temp_acorde_{nombre[:3]}.wav"
        with open(ruta, "wb") as f:
            f.write(data)
        RUTAS_ACORDES_WAV[nombre] = ruta
        
    print("Sintetizador de Acordes inicializado correctamente.")

def reproducir_sonido(ruta_wav):
    if WINSOUND_DISPONIBLE and ruta_wav:
        try:
            winsound.PlaySound(
                ruta_wav,
                winsound.SND_FILENAME | winsound.SND_ASYNC | winsound.SND_LOOP
            )
        except Exception as e:
            print(f"Error reproduciendo audio: {e}")

# ==========================================
# 2. RANGOS DE COLOR EN ESPACIO HSV
# ==========================================
# Definimos los rangos de color en HSV para cada ficha
# Saturación entre 80 - 255 para eliminar colores descoloridos y fondo blanco
# Brillo entre 80 - 255 para eliminar colores oscuros y sombras
# Rosa (para la ficha rosa) - cerca del 150
rosa_bajo = np.array([140, 100, 100])
rosa_alto = np.array([170, 255, 255])

# Azul (para la ficha azul) - cerca del 120
azul_bajo = np.array([95, 100, 100])
azul_alto = np.array([135, 255, 255])

# Verde (para la ficha verde) - cerca del 60
verde_bajo = np.array([40, 80, 80])
verde_alto = np.array([85, 255, 255])

# Amarillo (para la ficha amarilla) - cerca del 20
amarillo_bajo = np.array([20, 100, 100])
amarillo_alto = np.array([35, 255, 255])

# Umbral de área mínima para considerar que una ficha es visible
UMBRAL_AREA_MINIMA = 3000


def evaluar_visibilidad(hsv_frame, bajo, alto):
    # Evaluar si la ficha está visible en el frame actual
    mascara = cv2.inRange(hsv_frame, bajo, alto)
    # Aplicar operaciones morfológicas para reducir ruido y mejorar la detección
    mascara = cv2.erode(mascara, None, iterations=2)
    mascara = cv2.dilate(mascara, None, iterations=2)

    # Encontrar contornos de la máscara
    contornos, _ = cv2.findContours(mascara, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

    # Determinar el contorno con el área máxima
    area_maxima = 0
    contorno_max = None
    # Si hay contornos, encontrar el de mayor área
    if len(contornos) > 0:
        contorno_max = max(contornos, key=cv2.contourArea)
        area_maxima = cv2.contourArea(contorno_max)
    # Determinar si la ficha es visible según el área máxima encontrada    
    visible = area_maxima >= UMBRAL_AREA_MINIMA
    return visible, area_maxima, contorno_max

def main():
    if WINSOUND_DISPONIBLE:
        pregenerar_sonidos()
        
    camara = cv2.VideoCapture(2)
    
    if not camara.isOpened():
        print("Error: No es posible abrir la cámara web.")
        return

    print("=== SINTETIZADOR DE GESTOS Y ACORDES ACTIVO ===")
    print("Controles:")
    print(" - Pulsa 'M' para cambiar entre MODO NOTAS y MODO ACORDES")
    print(" - Pulsa 'q' o 'ESC' para salir.")

    modo_acordes = True  # Arranca en Modo Acordes 
    clave_actual = ""
    clave_anterior = None
    # Loop principal de captura de video y procesamiento
    while True:
        ret, frame = camara.read()
        if not ret:
            break
        # Suavizado del frame para reducir ruido y mejorar la detección de colores   
        frame_suavizado = cv2.blur(frame, (5, 5))
        hsv = cv2.cvtColor(frame_suavizado, cv2.COLOR_BGR2HSV)
        
        # Visibilidad
        rosa_visible, _, c_rosa = evaluar_visibilidad(hsv, rosa_bajo, rosa_alto)
        azul_visible, _, c_azul = evaluar_visibilidad(hsv, azul_bajo, azul_alto)
        verde_visible, _, c_verde = evaluar_visibilidad(hsv, verde_bajo, verde_alto)
        amarillo_visible, _, c_amarillo = evaluar_visibilidad(hsv, amarillo_bajo, amarillo_alto)
        
        rosa_tapado = not rosa_visible
        azul_tapado = not azul_visible
        verde_tapado = not verde_visible
        amarillo_tapado = not amarillo_visible
        
        # Lógica de combinaciones SIMÉTRICA para ambos modos
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
            clave_actual = "V (Sol Mayor)" if modo_acordes else "SOL"
            estado_texto = "Amarilla Tapada"
        else:
            clave_actual = "I (Do Mayor)" if modo_acordes else "DO"
            estado_texto = "Todas visibles"

        ruta_audio = RUTAS_ACORDES_WAV.get(clave_actual) if modo_acordes else RUTAS_NOTAS_WAV.get(clave_actual)

        # Reproducción sostenida cuando cambia de estado
        if clave_actual != clave_anterior:
            reproducir_sonido(ruta_audio)
            clave_anterior = clave_actual

        # Interfaz Gráfica 
        fichas = [
            ("ROSA", rosa_visible, c_rosa, (255, 0, 255)),
            ("AZUL", azul_visible, c_azul, (255, 0, 0)),
            ("VERDE", verde_visible, c_verde, (0, 255, 0)),
            ("AMARILLA", amarillo_visible, c_amarillo, (0, 255, 255))
        ]

        # Dibujar contornos y nombres de fichas detectadas
        for nombre, visible, contorno, color_bgr in fichas:
            if visible and contorno is not None:
                x, y, w, h = cv2.boundingRect(contorno)
                cv2.rectangle(frame, (x, y), (x + w, y + h), color_bgr, 2)
                cv2.putText(frame, nombre, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color_bgr, 2)

        # Panel 
        cv2.rectangle(frame, (20, 20), (520, 115), (0, 0, 0), -1)
        cv2.rectangle(frame, (20, 20), (520, 115), (255, 255, 255), 2)
        # Mostrar modo y nota/acorde actual
        modo_str = "MODO: ACORDES " if modo_acordes else "MODO: NOTAS INDIVIDUALES"
        cv2.putText(frame, modo_str, (30, 45), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
        cv2.putText(frame, f"SONANDO: {clave_actual}", (30, 80), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
        cv2.putText(frame, f"['M': Cambiar Modo | 'q': Salir] - {estado_texto}", (30, 105), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1)

        cv2.imshow("Acordes y Notas", frame)
        
        key = cv2.waitKey(1) & 0xFF
        if key in [ord('m'), ord('M')]:
            modo_acordes = not modo_acordes
            clave_anterior = None
            print(f"Modo cambiado a: {'ACORDES' if modo_acordes else 'NOTAS'}")
        elif key in [ord('q'), ord('Q'), 27]:
            print("Saliendo del programa...")
            break

    if WINSOUND_DISPONIBLE:
        winsound.PlaySound(None, winsound.SND_PURGE)
        
    camara.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()