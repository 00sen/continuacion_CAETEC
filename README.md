# Continuación CAETEC
## José Pablo A01275676
## Luis Arturo A01703572
### Despliegue de modelo para camas mediante script de python
### ________________________________________________________________
### 🧠 Requisitos
1. Python 3.8 o superior (compatible con 3.13)
2. Tener `pip` y acceso a una terminal
3. Instalar los paquetes necesarios:
pip install -r requirements.txt
⚠️ El paquete yolov5 ya incluye PyTorch (versión CPU).
Si quieres usar GPU, instala PyTorch manualmente con soporte CUDA antes de instalar yolov5.

### ▶️ Cómo usar el script
python script.py <carpeta_imagenes> [opciones]
Parámetros
input_folder: Carpeta con las imágenes a procesar
--weights: Ruta al archivo del modelo .pt (por defecto: model.pt)
--output: Carpeta donde se guardarán las imágenes clasificadas (por defecto: by_cow_count)
--device: Dispositivo de ejecución: cpu, cuda, etc. (por defecto: cpu)
--imgsz: Tamapo al que se redimensionan las imágenes antes de analizarlas (default: 640)

### 🖼️ Qué produce el script
Por cada imagen procesada:
- Se cuenta cuántas vacas hay.
- Se genera una versión bb_<nombre>.jpg de la imagen con los recuadros verdes dibujados

### ⚙️ Personalización
- Para cambiar los colores, grosor de los recuadros o texto, puedes modificar la función draw_boxes() en el archivo script.py

### 🧯 Manejo de errores
Si una imagen está dañada o no es reconocida, el script la omite automáticamente y continúa sin detenerse. Al final te reportará que imágenes fallaron.

### 📁 Estructura del proyecto
``` text
├── script.py              # Script principal
├── requirements.txt       # Librerías necesarias
├── model.pt               # Modelo entrenado (tú lo colocas)
├── imagenes/              # Carpeta con imágenes a analizar
└── by_cow_count/          # Resultado final clasificado por cantidad de vacas