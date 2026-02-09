import torch
import cv2
import numpy as np
from prueba_adaface import model, device, extraer_vector # Importamos tu modelo ya cargado

def calcular_similitud(vector1, vector2):
    # La similitud de coseno nos dice qué tan parecidos son dos rostros
    # 1.0 es la misma persona, valores cercanos a 0 son personas distintas
    similitud = torch.nn.functional.cosine_similarity(vector1, vector2)
    return similitud.item()

def identificar_persona(ruta_prueba, ruta_referencia):
    print(f"Comparando {ruta_prueba} con {ruta_referencia}...")
    
    # Extraemos los vectores de identidad de ambas fotos
    vec_prueba = extraer_vector(ruta_prueba)
    vec_ref = extraer_vector(ruta_referencia)
    
    if vec_prueba is not None and vec_ref is not None:
        puntaje = calcular_similitud(vec_prueba, vec_ref)
        
        # Umbral estándar SOTA: mayor a 0.4 suelen ser la misma persona
        if puntaje > 0.4:
            print(f"✅ ¡IDENTIDAD CONFIRMADA! Similitud: {puntaje:.4f}")
        else:
            print(f"❌ IDENTIDAD NO COINCIDE. Similitud: {puntaje:.4f}")
    else:
        print("No se pudo procesar alguna de las imágenes.")

# --- PRUEBA REAL ---
# Asegúrate de tener estas fotos en tu carpeta
# identificar_persona('foto_nueva.jpg', 'registros/yo.jpg')