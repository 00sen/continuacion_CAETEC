# Continuación CAETEC
## José Pablo A01275676
## Luis Arturo A01703572
### Despliegue de modelo para camas (y otros) mediante script de python
### ________________________________________________________________
### Información de modelos
- El modelo para el que se desarrolló el script.py fue para el modelo
modelBeds.py, sin embargo se puede cargar cualquier modelo que haya sido
entrenado con YOLOv5.
- El nuevo modelo entrenado, (el nuevo ángulo de la fila) puede ser cargado, se
  llama newModelRows.pt, en el csv serán impresos resultados de "que camas se
usan", resultado que puede ignorarse si se usa un modelo que no sea de camas.
De elegirse la opción images se dibujarán cajar verdes en las vacas ubicadas sin
ningún problema.
### 🧠 Requisitos
1. Python 3.9 o superior
2. Tener `pip` y acceso a una terminal
3. Instalar los paquetes necesarios:
pip install -r requirements.txt
⚠️ El paquete yolov5 ya incluye PyTorch (versión CPU).

### Ayuda con git
El proyecto completo se encuentra en el repositorio presente
- Si se quiere bajar se deberá tener instalado git
- Se podrá clonar de manera local utilizando el siguiente comando: git clone https://github.com/00sen/continuacion_CAETEC.git

### ▶️ Cómo usar el script
python script.py <carpeta_imagenes> [opciones]  
Parámetros opcionales:
- --model: Apunta al archivo del modelo .pt (por defecto: modelBeds.pt)
- --format: Resultado deseado. (images) si se quiere el resultado como imágenes con los recuadros pintados. (csv) si se quieren los resultados impresos en un csv. Por default (csv)       

Ejemplos de comando completo:

python script.py imagenes/ --model modelExample.pt --format images  
python script.py imagenes/ --model modelExample.pt --format csv  
python script.py imagenes/  

### 🖼️ Qué produce el script
Por cada imagen procesada:  
Si se eligió (csv) el cual es por defecto  
- Se cuenta cuántas vacas hay.  
- Se genera un csv con el nombre de la imagen, la cantidad de vacas y las camas
  que se están utilizando, de solo ser una cama será solo un número, de ser 2 o
más serán múltiples números entre comillas "".  

Si se eligió (images)  
- Se cuenta cuántas vacas hay.  
- Se genera una versión bb_<nombre>.jpg de la imagen con los recuadros verdes dibujados  

### ⚙️ Personalización
- Para cambiar los colores, grosor de los recuadros o texto, puedes modificar la función draw_boxes() en el archivo script.py
- Las camas se delimitan por una calibración pre definida en `script.py` línea 47 esta se puede ajustar para tomar en cuenta variaciones en la posición de la cámara
```python
DIVS_X = [260, 720, 1200]   # Pixeles de las camas de izquierda a derecha
```

### 🧯 Manejo de errores
Si una imagen está dañada o no es reconocida, el script la omite automáticamente y continúa sin detenerse. Al final te reportará que imágenes fallaron.

### 📁 Estructura del proyecto
``` text
├── script.py              # Script principal
├── requirements.txt       # Librerías necesarias
├── modelBeds.pt           # Modelo entrenado
├── imagenes/              # Carpeta con imágenes a analizar (no en el repo)
