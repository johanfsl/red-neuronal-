import cv2
import os
from pathlib import Path

def capturar_fotos_persona(nombre, num_fotos=50):
    """Captura fotos de una persona para entrenar el modelo"""
    
    # Crear carpeta si no existe
    ruta_persona = Path(f"dataset/personas/{nombre}")
    ruta_persona.mkdir(parents=True, exist_ok=True)
    
    # Inicializar cámara
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("❌ Error: No se puede abrir la cámara")
        return
    
    # Configurar resolución
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    
    contador = 0
    
    print(f"\n📸 Capturando {num_fotos} fotos de {nombre.upper()}")
    print("=" * 50)
    print("INSTRUCCIONES:")
    print("  - Presiona ESPACIO para capturar una foto")
    print("  - Presiona 'q' para salir")
    print("  - Mueve tu cara en diferentes ángulos")
    print("  - Cambia las expresiones faciales")
    print("=" * 50)
    
    while contador < num_fotos:
        ret, frame = cap.read()
        if not ret:
            print("❌ Error al leer frame")
            break
        
        # Crear copia para dibujar
        display_frame = frame.copy()
        
        # Mostrar información en pantalla
        cv2.putText(display_frame, f"Fotos capturadas: {contador}/{num_fotos}", 
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 
                    0.7, (0, 255, 0), 2)
        cv2.putText(display_frame, f"Persona: {nombre.upper()}", 
                    (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 
                    0.7, (255, 255, 0), 2)
        cv2.putText(display_frame, "Presiona ESPACIO para capturar", 
                    (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 
                    0.5, (255, 255, 255), 1)
        
        # Dibujar guía central
        h, w = display_frame.shape[:2]
        cv2.circle(display_frame, (w//2, h//2), 100, (0, 255, 0), 2)
        
        cv2.imshow('Captura de Rostros - AdaFace', display_frame)
        
        key = cv2.waitKey(1) & 0xFF
        
        # Capturar foto con ESPACIO
        if key == ord(' '):
            foto_path = ruta_persona / f"{nombre}_{contador:04d}.jpg"
            cv2.imwrite(str(foto_path), frame)
            print(f"✅ Foto {contador + 1}/{num_fotos} guardada: {foto_path.name}")
            contador += 1
            
            # Efecto visual de captura
            blank = frame.copy()
            blank.fill(255)
            cv2.imshow('Captura de Rostros - AdaFace', blank)
            cv2.waitKey(100)
        
        # Salir con 'q'
        elif key == ord('q'):
            print("\n⚠️  Captura cancelada por el usuario")
            break
    
    cap.release()
    cv2.destroyAllWindows()
    
    if contador > 0:
        print(f"\n✅ ¡COMPLETADO! {contador} fotos de {nombre.upper()} guardadas")
        print(f"📁 Ubicación: {ruta_persona}")
    else:
        print("\n❌ No se capturaron fotos")

if __name__ == "__main__":
    print("\n" + "="*50)
    print("  SISTEMA DE CAPTURA DE ROSTROS - AdaFace SOTA")
    print("="*50)
    
    # Pedir nombre
    nombre = input("\n👤 Nombre de la persona: ").strip().lower()
    
    if not nombre:
        print("❌ Error: Debes ingresar un nombre")
        exit(1)
    
    # Pedir número de fotos
    try:
        num_fotos = int(input("📸 ¿Cuántas fotos deseas capturar? (recomendado: 50-100): "))
        if num_fotos <= 0:
            raise ValueError
    except ValueError:
        print("❌ Error: Ingresa un número válido")
        exit(1)
    
    capturar_fotos_persona(nombre, num_fotos)

