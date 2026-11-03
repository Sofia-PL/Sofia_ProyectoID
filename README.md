# Sintetizador Gestual OpenCV + Loopstation

![Esquema del proyecto: webcam, pantalla y fichas de colores](docs/proyecto.png)

Instrumento musical controlado con visión artificial: la webcam observa fichas de colores y, según cuáles tapes con la mano, el programa toca notas o acordes. Además permite grabar bucles sincronizados con un metrónomo y exportar lo tocado como partitura y tablaturas.

## Índice

1. [Puesta en marcha](#1-puesta-en-marcha)
2. [De qué trata el proyecto](#2-de-qué-trata-el-proyecto)
3. [Modos, controles y combinaciones](#3-modos-controles-y-combinaciones)
4. [Guía de OpenCV (cv2)](#4-guía-de-opencv-cv2)
5. [Audio y todo lo que no es OpenCV](#5-audio-y-todo-lo-que-no-es-opencv)
6. [Estructura del código](#6-estructura-del-código)
7. [Ajustes y solución de problemas](#7-ajustes-y-solución-de-problemas)

---

## 1. Puesta en marcha

### Requisitos previos

Esta tabla recoge lo mínimo que necesitas tener antes de arrancar el proyecto. Las fichas pueden ser de cualquier material siempre que sus colores sean bien distinguibles.

| Requisito | Detalle |
|---|---|
| Python | 3.8 o superior |
| Webcam | Cualquier cámara (el programa prueba primero el índice 1 y, si no existe, el 0) |
| Fichas de colores | 4 fichas: rosa, azul, verde y roja (cartulinas, pegatinas, tapones...) |
| Sistema operativo | Windows, Linux o macOS (`winsound` solo existe en Windows, pero es opcional) |

### Estructura de carpetas

El proyecto se organiza en dos scripts y varias carpetas de datos. Las carpetas `audios/`, `imagenes/` y `tabs/` se crean solas al arrancar si no existen.

```text
proyectoID/
├── ReconocedorLoop.py                 # Programa principal (cámara + sintetizador + loopstation)
├── procesar_instrumento_grabacion.py  # Generador de muestras de audio desde vídeos .mp4
├── requirements.txt                   # Dependencias
├── .gitignore
├── audios/          # Muestras .wav (reales y sintéticas "temp_*")
├── grabaciones/     # Vídeos .mp4 de entrada para generar las muestras
├── imagenes/        # lamina_1 ... lamina_5 (tutorial) y tab_*.png (pianos de los acordes)
├── tabs/            # Partituras y tablaturas exportadas (tecla P)
└── docs/
    └── proyecto.png # Imagen de este README
```

Ajusta el nombre del segundo script si en tu proyecto se llama de otra forma.

### Instalación paso a paso

```bash
# 1. Clona o descarga el proyecto y entra en la carpeta
cd proyectoID

# 2. (Recomendado) Crea un entorno virtual
python -m venv venv

#    Windows
venv\Scripts\activate
#    Linux / macOS
source venv/bin/activate

# 3. Instala las dependencias
pip install -r requirements.txt
```

Si `requirements.txt` no incluye todo, instala las librerías a mano:

```bash
pip install opencv-python numpy pygame moviepy scipy
```

La siguiente tabla indica qué librerías son imprescindibles y cuáles solo amplían funciones. Sin `pygame` ni `winsound` el programa funciona, pero sin sonido.

| Librería | Obligatoria | Para qué se usa |
|---|:---:|---|
| `opencv-python` | Sí | Cámara, detección de color, HUD, ventanas, teclado y ratón |
| `numpy` | Sí | Arrays (las imágenes lo son) y síntesis de audio |
| `pygame` | No | Mezclador de audio multicanal (recomendado) |
| `moviepy` | No | Solo para el generador de muestras (extrae audio de `.mp4`) |
| `scipy` | No | Solo para el generador de muestras (lee y escribe `.wav`) |

### Ejecutar

```bash
python ReconocedorLoop.py
```

1. Se abre la ventana "Gesture Synth OpenCV + Loopstation" con el tutorial encima (se cierra con `ESC`, `ENTER`, `H` o la `[X]` roja).
2. Coloca las 4 fichas frente a la cámara, bien iluminadas y separadas entre sí.
3. Tapa y destapa fichas con la mano para tocar.

### Generar tus propias muestras de audio (opcional)

Si tienes grabaciones reales de violonchelo, piano u orquesta, el segundo script las convierte en muestras. Si no existen, el programa principal sintetiza sonidos por su cuenta, así que este paso no es necesario para probarlo.

```bash
# 1. Pon los vídeos en grabaciones/:
#    grabacion_cello.mp4, grabacion_piano_notas.mp4,
#    grabacion_piano_acordes.mp4, grabacion_orquesta_acordes.mp4
# 2. Ajusta los segundos de cada nota en la lista GRABACIONES del script
# 3. Ejecútalo:
python procesar_instrumento_grabacion.py
```

---

## 2. De qué trata el proyecto

El proyecto es un instrumento musical que se toca sin teclado ni ratón: la webcam observa cuatro fichas de colores y el programa deduce qué nota tocar según cuáles ve y cuáles están tapadas por tu mano.

Funcionamiento en cinco pasos:

1. La webcam captura el vídeo y lo refleja como un espejo.
2. OpenCV detecta, mediante máscaras de color en HSV, qué fichas se ven.
3. La combinación visible/tapada de las 4 fichas es un patrón que se traduce en una nota (DO, RE, MI...) o en un acorde (DoM, Rem, Mim...).
4. Esa nota o acorde se reproduce con el instrumento elegido.
5. Un metrónomo a 60 BPM actúa de reloj para grabar loops sincronizados y para exportar lo tocado como partitura y tablaturas.

### Resumen de funcionalidades

La tabla resume las capacidades principales del programa; cada una se detalla en las secciones siguientes.

| Funcionalidad | Qué hace |
|---|---|
| Detección de fichas | Máscaras HSV, morfología y contornos para saber qué ficha está visible |
| Modo notas / acordes | 7 notas de la escala de Do mayor o las 7 tríadas de esa escala |
| Instrumentos | Sintetizador, piano, violonchelo (notas) y orquesta (acordes) |
| Metrónomo | 60 BPM; la luz late siempre y el click es opcional |
| Loopstation | Graba lo que tocas, lo cuantiza a pulsos enteros y lo reproduce en bucle mientras tocas encima |
| Exportación | Pentagrama rítmico, tablatura de fichas y tablatura de piano (en `tabs/`) |
| Tutorial | 5 láminas superpuestas en la misma ventana, con botones y atajos |
| Brillo | Deslizador para compensar la luz de la habitación |

---

## 3. Modos, controles y combinaciones

### Controles de teclado

Todo el manejo del programa se hace con estas teclas, sin importar mayúsculas o minúsculas. Con el tutorial abierto el programa fuerza el silencio para que no suene nada por accidente.

| Tecla | Acción |
|:---:|---|
| `T` | Activa/desactiva el sonido del metrónomo (empieza desactivado; el reloj siempre corre) |
| `I` | Cambia de instrumento |
| `M` | Alterna modo acordes / notas |
| `R` | Loopstation: grabar, cerrar y cuantizar, y de nuevo grabar otro |
| `C` | Borra el loop |
| `P` | Empieza a grabar la partitura; al pulsar de nuevo exporta partitura y tablaturas |
| `H` | Abre/cierra el tutorial |
| `Q` / `ESC` | Salir (con el tutorial abierto, `ESC` solo lo cierra) |
| `A` `D` `←` `→` `Espacio` | Navegar por el tutorial (solo con el tutorial abierto) |

### Combinaciones de fichas

Cada nota o acorde corresponde a una combinación única de fichas visibles y tapadas. Cualquier combinación que no esté en la tabla produce silencio.

| Nota | Acorde | Rosa | Azul | Verde | Roja |
|:---:|:---|:---:|:---:|:---:|:---:|
| DO | I - Do Mayor (DoM) | Visible | Visible | Visible | Visible |
| RE | II - Re menor (Rem) | Tapada | Visible | Visible | Visible |
| MI | III - Mi menor (Mim) | Visible | Tapada | Visible | Visible |
| FA | IV - Fa Mayor (FaM) | Visible | Visible | Tapada | Visible |
| SOL | V - Sol Mayor (SolM) | Visible | Visible | Visible | Tapada |
| LA | VI - La menor (Lam) | Tapada | Tapada | Visible | Visible |
| SI | VII - Si disminuido (Sidim) | Tapada | Visible | Tapada | Visible |

### Modos e instrumentos

El modo decide si suena una sola frecuencia o una tríada, y con él cambia la lista de instrumentos que recorre la tecla `I`.

| Modo | Instrumentos disponibles | Qué suena |
|---|---|---|
| Acordes (por defecto) | Sintetizador, Piano, Orquesta | Una tríada (3 frecuencias a la vez) |
| Notas | Sintetizador, Piano, Violonchelo | Una sola frecuencia |

### Estados de la Loopstation

La loopstation es una pequeña máquina de estados que espera siempre al siguiente pulso del metrónomo para empezar a grabar, de modo que los bucles quedan acompasados.

```text
IDLE --R--> ARMED_REC --(siguiente pulso)--> RECORDING --R--> PLAYING
  ^            | R (cancela)                                     |
  +------------+------------------ C (borra) <-------------------+
                                    PLAYING --R--> ARMED_REC (nueva grabación)
```

La tabla describe qué ocurre en cada estado y qué mensaje muestra el HUD.

| Estado | Qué ocurre | Mensaje en el HUD |
|---|---|---|
| `IDLE` | Nada grabado | `pulsa R para grabar` |
| `ARMED_REC` | Espera el próximo pulso del metrónomo para empezar justo en el beat | `esperando pulso...` |
| `RECORDING` | Anota cada cambio de nota/instrumento con su tiempo relativo | `grabando X.Xs - R: cerrar` |
| `PLAYING` | La duración se cuantiza al nº entero de pulsos más cercano y el loop suena en bucle | `sonando Xs - C: borrar` |

---

## 4. Guía de OpenCV (cv2)

Esta sección recoge todas las funciones de OpenCV que usa el proyecto, agrupadas por funcionalidad, con cómo se usan y para qué sirven.

### Ideas básicas

Antes de las funciones conviene tener claras unas convenciones de OpenCV que explican muchas decisiones del código, como escribir el rojo como `(0, 0, 255)`.

| Concepto | Explicación |
|---|---|
| Imagen = array de NumPy | Forma `(alto, ancho, 3)`; cada píxel son 3 números de 0 a 255 |
| Orden BGR, no RGB | El rojo puro se escribe `(0, 0, 255)` |
| Coordenadas `(x, y)` | Origen `(0, 0)` arriba a la izquierda; `y` crece hacia abajo |
| Dibujo "en sitio" | Las funciones de dibujo modifican la misma variable que les pasas |
| `grosor = -1` | Rellena la figura |
| `cv2.LINE_AA` | Bordes suavizados (anti-aliasing) |

### 4.1 Cámara, ventana y entrada de usuario

Estas funciones abren la cámara, muestran el vídeo y recogen teclado y ratón. `waitKeyEx` es imprescindible: sin ella la ventana no se refresca.

| Función | Cómo se usa en el proyecto | Para qué sirve |
|---|---|---|
| `cv2.VideoCapture(n)` | `abrir_camara()` prueba los índices 1 y luego 0 | Abre la cámara número n |
| `camara.isOpened()` | Comprueba cada intento de apertura | Saber si la cámara se abrió bien |
| `camara.read()` | En el bucle principal: `ret, frame = camara.read()` | Captura un fotograma; `ret` indica si hubo éxito |
| `camara.release()` | Al salir y al descartar una cámara que no abre | Libera el recurso de la cámara |
| `cv2.flip(img, 1)` | `cv2.flip(frame, 1)` antes de procesar | Efecto espejo horizontal, más natural al moverte |
| `cv2.namedWindow(nombre)` | Crea la ventana "Gesture Synth OpenCV + Loopstation" | Necesaria para añadirle trackbar y ratón |
| `cv2.createTrackbar(...)` | Deslizador "Ajuste Brillo" (0 a 200, inicia en 100) | Compensar poca o mucha luz; llama a `actualizar_brillo_trackbar` |
| `cv2.setMouseCallback(...)` | Registra `Tutorial.al_hacer_clic` | Detectar clics en `[X]`, `<` y `>` del tutorial |
| `cv2.EVENT_LBUTTONDOWN` | Filtro dentro del callback del ratón | Reaccionar solo al clic izquierdo |
| `cv2.imshow(nombre, img)` | Vía `mostrar_imagen_segura()` | Muestra el vídeo y las ventanas de exportación |
| `cv2.waitKeyEx(1)` | Una vez por fotograma | Lee la tecla y refresca la ventana; la versión Ex reconoce las flechas |
| `cv2.destroyAllWindows()` | Al terminar | Cierra todas las ventanas de OpenCV |

### 4.2 Visión artificial: detección de fichas

Este es el núcleo del proyecto: se filtra cada color en HSV, se limpia la máscara y se mide la mancha mayor para decidir si la ficha está visible.

| Función | Cómo se usa en el proyecto | Para qué sirve |
|---|---|---|
| `cv2.blur(img, (5, 5))` | Desenfoque 5x5 antes de pasar a HSV | Quita ruido de la cámara y estabiliza las máscaras |
| `cv2.cvtColor(img, COLOR_BGR2HSV)` | En `detectar_fichas` y `ajustar_brillo_frame` | HSV separa color (H) de intensidad (S, V): detectar un color es más estable con luz cambiante |
| `cv2.cvtColor(img, COLOR_HSV2BGR)` | Al terminar el ajuste de brillo | Volver a BGR para dibujar y mostrar |
| `cv2.split(img)` | Separa H, S y V | Modificar solo el brillo (V) |
| `cv2.merge((h, s, v))` | Reúne los tres canales | Reconstruir la imagen tras tocar V |
| `cv2.inRange(hsv, bajo, alto)` | Una llamada por rango de color de cada ficha | Crea una máscara blanco/negro: blanco = píxel dentro del rango |
| `cv2.bitwise_or(a, b)` | Une las dos máscaras del color rojo | El rojo está en ambos extremos del círculo de matiz (0 y 180) |
| `cv2.erode(m, None, iterations=2)` | Primera mitad de la apertura morfológica | Encoge las manchas y borra puntitos de ruido |
| `cv2.dilate(m, None, iterations=2)` | Segunda mitad de la apertura | Devuelve su tamaño a las manchas que sobrevivieron |
| `cv2.findContours(m, RETR_LIST, CHAIN_APPROX_SIMPLE)` | Sobre la máscara limpia | Encuentra los contornos de cada mancha, guardando solo los vértices |
| `cv2.contourArea(c)` | `max(contornos, key=cv2.contourArea)` | Elige la mancha mayor y mide su área |
| `cv2.boundingRect(c)` | En `dibujar()` | Rectángulo `(x, y, ancho, alto)` para enmarcar la ficha |

Regla de visibilidad: una ficha está visible si la mayor mancha de su color tiene un área igual o superior a `UMBRAL_AREA_MINIMA` (3000 px). Si tu mano la tapa, la mancha desaparece o baja del umbral y se considera tapada.

#### Rangos HSV de cada ficha

Cada ficha se define por un rango de matiz (H) y se exigen saturación y brillo mínimos de 50 para descartar grises y zonas oscuras. En OpenCV, H va de 0 a 180 y S y V de 0 a 255.

| Ficha | H mín - máx | S | V | Color del recuadro (BGR) |
|---|:---:|:---:|:---:|:---:|
| Rosa | 138 - 168 | 50 - 255 | 50 - 255 | `(255, 0, 255)` |
| Azul | 95 - 130 | 50 - 255 | 50 - 255 | `(255, 0, 0)` |
| Verde | 35 - 85 | 50 - 255 | 50 - 255 | `(0, 255, 0)` |
| Roja | 0 - 8 y 172 - 180 | 50 - 255 | 50 - 255 | `(0, 0, 255)` |

### 4.3 Dibujo del HUD, tarjetas y partitura

Todo lo que se ve en pantalla, desde el panel azul hasta las notas del pentagrama, está dibujado con primitivas de OpenCV, sin imágenes externas.

| Función | Cómo se usa en el proyecto | Para qué sirve |
|---|---|---|
| `cv2.rectangle` | Recuadros de fichas, teclado de piano, barras, botón `[X]`, tarjetas | Rectángulos (rellenos con -1) |
| `cv2.circle` | Fichas de las tarjetas, luz del metrónomo y su halo | Círculos |
| `cv2.line` | Divisiones del HUD, X roja de fichas tapadas, líneas del pentagrama, plicas | Líneas rectas |
| `cv2.ellipse` | Cabezas de las notas del pentagrama (elipse inclinada 30 grados) | Elipses; rellena = negra/corchea, hueca = blanca/redonda |
| `cv2.polylines` | Marcos biselados del HUD y marco neón | Polilínea cerrada (contorno) |
| `cv2.fillPoly` | Fondo de los paneles translúcidos | Polígono relleno |
| `cv2.putText` | Todos los textos (HUD, tarjetas, partitura, tutorial) | Escribe texto (fuente Hershey: no dibuja tildes ni ñ) |
| `cv2.getTextSize` | `texto_centrado`, `escala_ajustada` y barra del tutorial | Mide un texto para centrarlo o reducirlo hasta que quepa |
| `cv2.addWeighted` | `panel_translucido`, `marco_neon`, halo del metrónomo | Mezcla con transparencia (`a*alfa + b*(1-alfa)`): paneles de cristal y resplandor neón |
| `cv2.FONT_HERSHEY_SIMPLEX` | Constante `FUENTE` | Fuente única del proyecto |

Los efectos visuales del HUD se construyen combinando varias de estas funciones. La tabla explica cómo se consigue cada uno.

| Efecto | Cómo se consigue |
|---|---|
| Panel translúcido biselado | `fillPoly` sobre una copia y `addWeighted` con `alpha = 0.80` |
| Marco neón al sonar una nota | `polylines` muy gruesa en una copia, `addWeighted` y `polylines` nítida encima; la intensidad decae con `exp(-edad*6)` |
| Luz del metrónomo que late | `circle` cuyo radio pasa de 28 px (pulso) a 12 px (reposo) y cuyo color se mezcla del amarillo al fondo |
| Texto que se ajusta | `getTextSize` en bucle reduciendo la escala hasta que cabe |

### 4.4 Imágenes en disco y transformaciones

Estas funciones leen y guardan imágenes y las redimensionan. Se usan `imdecode` e `imencode` en lugar de `imread` e `imwrite` para que funcionen rutas con tildes en Windows.

| Función | Cómo se usa en el proyecto | Para qué sirve |
|---|---|---|
| `cv2.imdecode(bytes, IMREAD_COLOR)` | `leer_imagen_segura()` junto a `np.fromfile` | Leer imágenes con rutas con tildes en Windows |
| `cv2.imencode(".png", img)` | `guardar_imagen()` junto a `buf.tofile` | Guardar PNG con rutas con tildes |
| `cv2.resize(img, (ancho, alto))` | Imágenes de piano en las tarjetas y láminas del tutorial | Cambiar de tamaño |
| `cv2.INTER_AREA` | Láminas del tutorial | Interpolación recomendada al reducir (más nítida) |

### 4.5 Dónde se usa cada grupo

Esta tabla sirve de mapa para localizar en el código cada bloque de funcionalidad visual.

| Grupo | Funciones del código |
|---|---|
| Cámara y ventana | `abrir_camara()`, `main()` |
| Brillo | `ajustar_brillo_frame()`, `actualizar_brillo_trackbar()` |
| Detección | `evaluar_visibilidad()`, `detectar_fichas()` |
| Tarjetas de exportación | `generar_tarjeta_fichas()`, `generar_tarjeta_piano()`, `dibujar_piano_esquematico()` |
| Partitura | `generar_partitura_pentagrama_ritmica()`, `montar_cuadricula()` |
| HUD | `dibujar_hud()`, `dibujar_metronomo()`, `panel_translucido()`, `marco_biselado()`, `marco_neon()` |
| Tutorial | clase `Tutorial` (`dibujar`, `al_hacer_clic`, `manejar_tecla`) |

---

## 5. Audio y todo lo que no es OpenCV

### 5.1 Cómo se genera el sonido (NumPy y síntesis aditiva)

Cada nota es una frecuencia en Hz y un acorde son tres frecuencias a la vez. Un sonido se construye sumando ondas senoidales (la fundamental y sus armónicos, que son múltiplos de su frecuencia) y multiplicando el resultado por una envolvente, que es la curva de volumen en el tiempo. NumPy calcula las 44 100 muestras por segundo de golpe, sin bucles.

La tabla compara los cuatro estilos de sonido sintético que genera el programa.

| Estilo | Armónicos | Envolvente | Carácter |
|---|---|---|---|
| METRONOMO | Seno puro de 1000 Hz | Decae linealmente de 1 a 0 (50 ms) | Click corto |
| PIANO | f + 0,45 (2f) + 0,20 (3f) + 0,10 (4f) | Ataque de 5 ms y caída exponencial `exp(-3.5*t)` | Percusivo, se apaga solo |
| ORQUESTA | 5 senos con voces desafinadas (x1,003 y x0,997), octava arriba y abajo | Ataque y relajación de 120 ms | Coro de cuerdas con vibrato de 5 Hz |
| SINTETIZADOR | 0,65 (f) + 0,35 (2f) | Fade-in y fade-out de 80 ms | Limpio, sin clicks |

Los detalles técnicos del formato de los archivos generados son los siguientes.

| Detalle | Valor / explicación |
|---|---|
| Formato | WAV mono, 16 bits, 44 100 Hz (calidad CD) |
| Amplitud | 16 000 (el límite de 16 bits es 32 767); 24 000 para el click |
| Acordes | Se divide entre el nº de notas para que no saturen |
| Conversión | `np.clip(...).astype('<i2')` (little-endian de 16 bits) |
| Cabecera WAV | Escrita a mano con `struct.pack('<4sI4s4sIHHIIHH4sI', ...)` (44 bytes: `RIFF`, `fmt `, `data`...) |

### 5.2 Librerías de reproducción

El programa usa `pygame.mixer` como reproductor preferido y `winsound` como alternativa solo para Windows. Ambas son opcionales: si no hay ninguna, el programa sigue funcionando pero sin sonido.

| Librería | Papel | Ventajas / limitaciones |
|---|---|---|
| `pygame.mixer` | Reproductor preferido | Varios canales a la vez, baja latencia (buffer 512) |
| `winsound` | Alternativa solo Windows | Un único canal; `Beep` para el click y `PlaySound` para el resto |

La tabla siguiente detalla las funciones concretas de audio que se utilizan.

| Función | Qué hace en el proyecto |
|---|---|
| `pygame.mixer.pre_init(44100, -16, 1, 512)` | Configura frecuencia, 16 bits con signo, mono y buffer pequeño (poca latencia) |
| `pygame.mixer.init()` | Inicia el mezclador |
| `pygame.mixer.set_num_channels(8)` | Hasta 8 sonidos simultáneos |
| `pygame.mixer.Channel(0)` / `Channel(1)` | Canal 0 = loop grabado, canal 1 = lo que tocas en vivo |
| `pygame.mixer.Sound(ruta)` | Carga el `.wav` en memoria listo para sonar sin retardo |
| `canal.play(sonido, loops=-1)` | Reproduce en bucle hasta que se detenga |
| `canal.stop()` | Detiene el canal |
| `winsound.Beep(1000, 40)` | Click del metrónomo (reserva) |
| `winsound.PlaySound(ruta, SND_FILENAME \| SND_ASYNC \| SND_LOOP)` | Reproduce un archivo en bucle (reserva) |

### 5.3 Dónde busca los sonidos (muestras reales o sintéticas)

Para cada nota o acorde, `cargar_sonido()` busca primero el archivo real en `audios/`. Si no existe, genera uno sintético y lo guarda con el prefijo `temp_`.

| Instrumento | Tipo | Archivo buscado en `audios/` |
|---|---|---|
| Sintetizador | Notas | `synth_nota_DO.wav` ... |
| Sintetizador | Acordes | `synth_acorde_I_DoM.wav` ... |
| Piano | Notas | `piano_nota_DO.wav` ... |
| Piano | Acordes | `piano_acorde_I_DoM.wav` ... |
| Violonchelo | Notas | `cello_DO.wav` ... |
| Orquesta | Acordes | `orquesta_acorde_I_DoM.wav` ... |
| Metrónomo | Click | `temp_metronomo_click.wav` (siempre sintético) |

Si un instrumento no tiene un sonido concreto, `obtener_recurso_audio()` usa el sintetizador como reserva. Las etiquetas de acordes en los nombres de archivo son `I_DoM`, `II_Rem`, `III_Mim`, `IV_FaM`, `V_SolM`, `VI_Lam` y `VII_Sidim`.

### 5.4 Canales: directo y loop

Hay dos canales reservados para que lo que tocas en vivo no corte el bucle grabado y viceversa.

| Canal | Contenido | Se detiene cuando... |
|---|---|---|
| `vivo` (1) | Lo que tocas ahora mismo | Pasas a una combinación inválida (silencio) o cambias de modo |
| `loop` (0) | El bucle grabado | Pulsas `C` o grabas uno nuevo |

### 5.5 Metrónomo y sincronización

El metrónomo es el reloj común de todo el programa: marca un pulso por segundo y la loopstation se apoya en él para empezar y terminar siempre en un beat.

| Concepto | Cómo funciona |
|---|---|
| Reloj | 60 BPM, un pulso por segundo (`intervalo = 60 / bpm`), basado en `time.time()` |
| Sonido | Solo suena si lo activas con `T` (el reloj y la luz corren siempre) |
| Inicio de grabación | `R` arma la loopstation y la grabación empieza exactamente en el siguiente pulso |
| Cuantización | Al cerrar, `beats = max(1, round(duración / intervalo))`: el loop dura un nº entero de pulsos |
| Reproducción | `t = (ahora - t_repro) % duración` da la posición dentro del bucle; al pasar de un evento al siguiente cambia el sonido |
| Alineación | `t_repro` se alinea con el último pulso del metrónomo para que el loop encaje en el ritmo |

### 5.6 Interpretación del gesto (anti-rebote)

Para que un parpadeo de la cámara no genere notas falsas, el gesto debe mantenerse estable `TIEMPO_MINIMO_ESTABILIDAD = 0.18 s` antes de aceptarse. Solo entonces se confirma la nota, se reproduce y se registra en el loop y en la partitura.

### 5.7 Exportación de partitura y tablaturas (tecla P)

La primera pulsación de `P` empieza a grabar y la segunda cierra la última nota y exporta tres imágenes PNG en `tabs/`, con fecha y hora (`YYYYMMDD_HHMMSS`) para no sobrescribir exportaciones anteriores.

| Archivo | Contenido |
|---|---|
| `tablatura_fichas_<marca>.png` | Tarjeta por nota con círculos de color (visibles) o gris con X roja (tapadas) |
| `tablatura_piano_<marca>.png` | Imagen del piano de cada acorde (de `imagenes/tab_*.png` o un teclado esquemático de reserva) |
| `partitura_ritmica_<marca>.png` | Pentagrama en clave de Sol, 8 notas por sistema |

En el pentagrama, la figura musical depende de cuánto tiempo mantienes la nota. Las notas que duran menos de `UMBRAL_MINIMO_RECORD_NOTA = 0.25 s` no entran en la partitura.

| Tiempo sostenido | Figura |
|---|---|
| 3 s o más | Redonda (sin plica) |
| 1,5 a 3 s | Blanca (cabeza hueca con plica) |
| 0,75 a 1,5 s | Negra (cabeza rellena con plica) |
| Menos de 0,75 s | Corchea (negra con corchete) |

### 5.8 Generador de muestras (`procesar_instrumento_grabacion.py`)

Este script convierte grabaciones reales `.mp4` en muestras `.wav` recortadas a la misma duración. Para cada nota extrae 1,0 s desde su marca de tiempo, aplica un fade-in y fade-out de 80 ms para evitar clicks y guarda `prefijo_ETIQUETA.wav` en `audios/`.

| Librería | Uso |
|---|---|
| `moviepy` (`AudioFileClip`) | Extrae la pista de audio del vídeo (44,1 kHz, mono, 16 bits) |
| `scipy.io.wavfile` | Lee el audio extraído y escribe cada muestra recortada |
| `numpy` | Recorte por índices, relleno con silencio y envolvente |
| `tempfile` | Guarda el audio completo temporalmente y lo borra al acabar |

La siguiente tabla muestra qué vídeo genera qué muestras. Para ajustar los cortes, edita los segundos de `"tiempos"` de cada grabación en la lista `GRABACIONES`.

| Vídeo de entrada | Prefijo de salida | Contenido |
|---|---|---|
| `grabacion_cello.mp4` | `cello` | 7 notas de violonchelo |
| `grabacion_piano_notas.mp4` | `piano_nota` | 7 notas de piano |
| `grabacion_piano_acordes.mp4` | `piano_acorde` | 7 acordes de piano |
| `grabacion_orquesta_acordes.mp4` | `orquesta_acorde` | 7 acordes de orquesta |

### 5.9 Otras librerías estándar de Python

Además de las librerías externas, el proyecto usa cuatro módulos de la biblioteca estándar para tareas auxiliares.

| Librería | Uso |
|---|---|
| `struct` | Escribir a mano la cabecera de los `.wav` |
| `math` | `ceil` (nº de pentagramas) y `exp` (decaimiento del resplandor) |
| `time` | Reloj del metrónomo, duración de notas, estabilidad del gesto y marcas de fecha |
| `os` | Rutas relativas al script y creación de carpetas |

---

## 6. Estructura del código

El código se organiza en clases pequeñas con una responsabilidad cada una, y la clase `SintetizadorGestual` las une para que `main()` solo tenga que llamar a `procesar()`, `dibujar()` y `tecla()`.

| Clase / módulo | Responsabilidad |
|---|---|
| `Metronomo` | Reloj de 60 BPM y estado del click |
| `Interprete` | Convierte el patrón de fichas en nota/acorde estable; gestiona modo e instrumento |
| `Loopstation` | Máquina de estados `IDLE`, `ARMED_REC`, `RECORDING`, `PLAYING` |
| `GrabadorPartitura` | Acumula notas tocadas y dispara la exportación |
| `Tutorial` | Láminas superpuestas, botones y clics |
| `SintetizadorGestual` | Une todo: `procesar()`, `dibujar()` y `tecla()` |
| `main()` | Abre cámara y ventana, y ejecuta el bucle principal |

---

## 7. Ajustes y solución de problemas

### Constantes que puedes tocar

Estas constantes controlan la sensibilidad de la detección y del anti-rebote; son lo primero que conviene ajustar si el reconocimiento no funciona bien en tu entorno.

| Constante | Valor | Efecto |
|---|:---:|---|
| `UMBRAL_AREA_MINIMA` | 3000 | Área mínima (px) para considerar una ficha visible. Súbela si detecta fichas fantasma; bájala si las pierde |
| `TIEMPO_MINIMO_ESTABILIDAD` | 0.18 | Segundos que debe mantenerse un gesto. Más alto = menos errores pero más retardo |
| `UMBRAL_MINIMO_RECORD_NOTA` | 0.25 | Las notas más cortas no entran en la partitura |
| `FICHAS` (rangos HSV) | ver tabla | Ajusta H, S y V si tus fichas son de otro tono |
| `Metronomo(bpm=60)` | 60 | Tempo del reloj |

### Problemas frecuentes

La mayoría de fallos se deben a la iluminación, a la cámara elegida o a que falta alguna librería de audio.

| Problema | Posible causa y solución |
|---|---|
| `Error: No es posible abrir la cámara web` | Cámara en uso por otra aplicación o sin permisos. Cierra otros programas |
| Se abre la cámara equivocada | El programa prueba primero el índice 1; cambia el orden `(1, 0)` en `abrir_camara()` |
| Detecta fichas que no están o no detecta las reales | Mejora la luz, usa el deslizador "Ajuste Brillo" o reajusta los rangos HSV |
| El rojo falla con la piel o la ropa | Aleja las fichas de tonos parecidos o estrecha el rango rojo |
| No suena nada | Instala `pygame` (`pip install pygame`) y comprueba el volumen. Con el tutorial abierto no suena nada a propósito |
| Las notas "tartamudean" | Sube `TIEMPO_MINIMO_ESTABILIDAD` y evita sombras sobre las fichas |
| No aparecen tildes en pantalla | Es normal: `cv2.putText` no dibuja tildes ni ñ |
| Las láminas del tutorial salen en texto simple | Faltan `lamina_1` ... `lamina_5` en `imagenes/`; el programa usa un texto de reserva |
| El generador dice que no encuentra un `.mp4` | Colócalo en `grabaciones/` con el nombre exacto; si falta, simplemente se omite |
