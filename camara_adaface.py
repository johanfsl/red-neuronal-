import torch
import cv2
import numpy as np
import sys
import os
from prueba_adaface import model, device # Importamos tu modelo ya listo

# 1. Función para preprocesar lo que ve la cámara
def preprocesar_frame(face_crop):
    face_crop = cv2.resize(face_crop, (112, 112))
    face_crop = cv2.cvtColor(face_crop, cv2.COLOR_BGR2RGB)
    face_crop = (face_crop.astype(np.float32) - 127.5) / 128.0
    face_crop = np.transpose(face_crop, (2, 0, 1))
    
    # IMPORTANTE: Añadimos .half() para que coincida con el modelo MS1MV3
    tensor = torch.from_numpy(face_crop).unsqueeze(0).to(device).half() 
    return tensor

# 2. Cargar tu foto de referencia (Base de Datos)
ruta_yo = 'registros/yo.jpg'
img_yo = cv2.imread(ruta_yo)
if img_yo is None:
    print(f"❌ Error: Pon tu foto en {ruta_yo}")
    sys.exit()

# Generar el vector (embedding) de "Yo"
with torch.no_grad():
    tensor_yo = preprocesar_frame(img_yo)
    vec_referencia = model(tensor_yo)

# 3. Iniciar Cámara Web
cap = cv2.VideoCapture(0)
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

print("🚀 Cámara iniciada. Presiona 'q' para salir.")

while True:
    ret, frame = cap.read()
    if not ret: break
    
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.1, 4)
    
    for (x, y, w, h) in faces:
        # Extraer solo el rostro
        rostro_actual = frame[y:y+h, x:x+w]
        
        with torch.no_grad():
            tensor_actual = preprocesar_frame(rostro_actual)
            vec_actual = model(tensor_actual)
            
            # Comparar similitud de coseno
            similitud = torch.nn.functional.cosine_similarity(vec_actual, vec_referencia).item()
        
        # Umbral: Si es mayor a 0.45, eres tú
        label = f"Identificado: {similitud:.2f}" if similitud > 0.45 else "Desconocido"
        color = (0, 255, 0) if similitud > 0.45 else (0, 0, 255)
        
        cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
        cv2.putText(frame, label, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    cv2.imshow('Reconocimiento AdaFace SOTA', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'): break

cap.release()
cv2.destroyAllWindows()