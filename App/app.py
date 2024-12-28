from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from fastapi import Request
from fastapi.middleware.cors import CORSMiddleware

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
import cv2
import base64
import numpy as np
import math

app = FastAPI()

# Configuración CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Montar archivos estáticos
app.mount("/assets", StaticFiles(directory="templates/assets"), name="assets")

# Configuración de plantillas HTML
templates = Jinja2Templates(directory="templates")

# Rutas para servir tus archivos HTML (sirve para ingresar a mis diferentes archivo html responsives
@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request):
    #Aqui se coloca la ruta, lo que no es lo mismo que el nombr de ruta
    return templates.TemplateResponse("login.html", {"request": request})

# WebSocket para transmitir video, modelo basado en:https://www.youtube.com/watch?v=oEbkFPIMkjo
@app.websocket("/ws/video")
async def video_stream(websocket: WebSocket):
    await websocket.accept()
    capture = cv2.VideoCapture(0)  # Abre la cámara por defecto

    while True:
        min_contour_area = 3000  # Área mínima para considerar un contorno

        ret, frame = capture.read()
        if not ret:
            break  # Si no se puede leer el frame, sale del bucle
        
        # Voltea el frame horizontalmente para mejorar el alineamiento
        frame = cv2.flip(frame, 1)

        # Dibuja un rectángulo más grande para el ROI
        cv2.rectangle(frame, (50, 50), (600, 600), (0, 0, 0), 2)  # Marco más grande y visible

        # Recorta la imagen con las nuevas coordenadas (más grande)
        crop_image = frame[50:600, 50:600]
        # Aplica desenfoque Gaussiano para suavizar el ruido
        blur = cv2.GaussianBlur(crop_image, (3, 3), 0)

        # Convierte a espacio de color HSV
        hsv = cv2.cvtColor(blur, cv2.COLOR_BGR2HSV)

        # Ajustar los valores de HSV para una mejor detección de la piel
        lower_skin = np.array([0, 20, 70])
        upper_skin = np.array([20, 255, 255])

        # Crear una máscara basada en el rango HSV
        mask = cv2.inRange(hsv, lower_skin, upper_skin)

        # Aumenta el tamaño del kernel para las operaciones morfológicas
        kernel = np.ones((5, 5), np.uint8)  # Aumenta el tamaño del kernel

        # Aplicar transformaciones morfológicas para filtrar el ruido de fondo
        dilation = cv2.dilate(mask, kernel, iterations=1)
        erosion = cv2.erode(dilation, kernel, iterations=1)    

        # Aplicar un desenfoque Gaussiano y umbral
        filtered = cv2.GaussianBlur(erosion, (3, 3), 0)
        ret, thresh = cv2.threshold(filtered, 100, 255, 0)

        # Encontrar contornos
        contours, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        try:
            # Filtrar los contornos basados en el área mínima
            contours = [c for c in contours if cv2.contourArea(c) > min_contour_area]

            if len(contours) == 0:
                raise ValueError("No contours found.")

            # Encontrar el contorno más grande según el área
            contour = max(contours, key=lambda x: cv2.contourArea(x))
            area = cv2.contourArea(contour)

            # Ignorar contornos pequeños que probablemente son ruido
            if area < 5000:
                raise ValueError("Contour too small.")

            # Crear un rectángulo de contorno alrededor del objeto
            x, y, w, h = cv2.boundingRect(contour)
            cv2.rectangle(crop_image, (x, y), (x + w, y + h), (0, 0, 100), 2)

            # Encontrar el casco convexo
            hull = cv2.convexHull(contour)

            # Dibujar contornos y el casco
            drawing = np.zeros(crop_image.shape, np.uint8)
            cv2.drawContours(drawing, [contour], -1, (0, 0, 100), 2)
            cv2.drawContours(drawing, [hull], -1, (0, 0, 255), 2)

            # Defectos de convexidad
            hull = cv2.convexHull(contour, returnPoints=False)
            defects = cv2.convexityDefects(contour, hull)

            count_defects = 0
            if defects is not None:
                for i in range(defects.shape[0]):
                    s, e, f, d = defects[i, 0]
                    start = tuple(contour[s][0])
                    end = tuple(contour[e][0])
                    far = tuple(contour[f][0])

                    # Calcular el ángulo
                    a = math.sqrt((end[0] - start[0])**2 + (end[1] - start[1])**2)
                    b = math.sqrt((far[0] - start[0])**2 + (far[1] - start[1])**2)
                    c = math.sqrt((end[0] - far[0])**2 + (end[1] - far[1])**2)
                    angle = math.acos((b**2 + c**2 - a**2) / (2 * b * c)) * 57

                    # Si el ángulo es menor a 90, es un defecto
                    if angle <= 90:
                        count_defects += 1
                        cv2.circle(crop_image, far, 4, [255, 0, 0], -1)

                    cv2.line(crop_image, start, end, [0, 255, 0], 2)

            # Interpretación del gesto
            if count_defects == 0:
                cv2.putText(frame, "UNO", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 0), 3)
            elif count_defects == 1:
                cv2.putText(frame, "DOS", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 0), 3)
            elif count_defects == 2:
                cv2.putText(frame, "TRES", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 0), 3)
            elif count_defects == 3:
                cv2.putText(frame, "CUATRO", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 0), 3)
            elif count_defects == 4:
                cv2.putText(frame, "CINCO", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 0), 3)

        except ValueError as e:
            print(f"Warning: {e}")
            drawing = np.zeros(crop_image.shape, np.uint8)

        # Convertir el frame a JPEG
        _, buffer = cv2.imencode(".jpg", frame)
        byte_data = base64.b64encode(buffer).decode('utf-8')
        
        # Enviar el frame como una cadena base64 al cliente
        await websocket.send_text(byte_data)

    capture.release()

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)
