import cv2
import torch
import torch.nn.functional as F
import numpy as np
import pickle
import os
import urllib.request
from pathlib import Path
from model import common
from datetime import datetime
import json
import time

# --- CONFIGURACIÓN DE MODELOS ---
PROTOTXT = "deploy.prototxt"
MODEL_FILE = "res10_300x300_ssd_iter_140000.caffemodel"

def descargar_modelos():
    urls = {
        PROTOTXT: "https://raw.githubusercontent.com/opencv/opencv/master/samples/dnn/face_detector/deploy.prototxt",
        MODEL_FILE: "https://github.com/opencv/opencv_3rdparty/raw/dnn_samples_face_detector_20170830/res10_300x300_ssd_iter_140000.caffemodel"
    }
    for archivo, url in urls.items():
        if not os.path.exists(archivo):
            print(f"📥 Descargando {archivo}...")
            urllib.request.urlretrieve(url, archivo)

class ReconocedorDestroyeGPU:
    def __init__(self, model_path='weights/adaface_ir101_ms1mv3.ckpt'):
        descargar_modelos()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.db_path = 'identidades_nuclear.pkl'
        
        # --- PARÁMETROS HARDCORE ---
        self.threshold = 0.68
        self.fotos_registro = 1500
        self.batch_size = 256
        self.vectores_por_foto = 8
        
        print(f"🔥 MODO DESTRUCCIÓN: {self.device} | FP16 + Multi-Scale")
        
        # Cargar AdaFace
        self.model = common.iresnet101(num_features=512).to(self.device).eval().half()
        try:
            checkpoint = torch.load(model_path, map_location=self.device, weights_only=True)
            state_dict = checkpoint.get('state_dict', checkpoint)
            self.model.load_state_dict({k.replace('model.', ''): v for k, v in state_dict.items()}, strict=False)
            print("✅ AdaFace IR101 cargado.")
        except Exception as e:
            print(f"❌ Error: {e}")

        # Detector SSD
        self.net = cv2.dnn.readNetFromCaffe(PROTOTXT, MODEL_FILE)
        self.net.setPreferableBackend(cv2.dnn.DNN_BACKEND_DEFAULT)
        self.net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
        
        self.db_nombres = []
        self.db_embeddings = None 
        self.cargar_base_datos()

    def normalizar_iluminacion(self, img):
        """Normalización CLAHE para anti-luz"""
        lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
        l = clahe.apply(l)
        lab = cv2.merge([l, a, b])
        return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)

    def augmentar_imagen(self, img):
        """Augmentación manual sin albumentations"""
        # Variación 1: Brillo
        img1 = cv2.convertScaleAbs(img, alpha=1.3, beta=30)
        
        # Variación 2: Oscurecer
        img2 = cv2.convertScaleAbs(img, alpha=0.7, beta=-30)
        
        # Variación 3: Blur
        img3 = cv2.GaussianBlur(img, (5, 5), 0)
        
        # Variación 4: Sharpen
        kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
        img4 = cv2.filter2D(img, -1, kernel)
        
        # Variación 5: Contraste alto
        img5 = cv2.convertScaleAbs(img, alpha=1.5, beta=0)
        
        # Variación 6: Gamma correction
        gamma = 1.5
        invGamma = 1.0 / gamma
        table = np.array([((i / 255.0) ** invGamma) * 255 for i in range(256)]).astype("uint8")
        img6 = cv2.LUT(img, table)
        
        # Variación 7: HSV shift
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        hsv[:,:,2] = cv2.add(hsv[:,:,2], 20)
        img7 = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
        
        return [img, img1, img2, img3, img4, img5, img6, img7]

    def cargar_base_datos(self):
        """Carga vectores"""
        if Path(self.db_path).exists():
            with open(self.db_path, 'rb') as f:
                data = pickle.load(f)
            
            nombres, todos_vectores = [], []
            for nombre, lista_embs in data.items():
                todos_vectores.extend(lista_embs)
                nombres.extend([nombre] * len(lista_embs))
            
            if todos_vectores:
                self.db_embeddings = torch.tensor(np.array(todos_vectores)).to(self.device).half()
                self.db_embeddings = F.normalize(self.db_embeddings, p=2, dim=1)
                self.db_nombres = nombres
                print(f"🧠 DB: {len(set(nombres))} personas | {len(todos_vectores):,} vectores")

    @torch.no_grad()
    def procesar_lote_hardcore(self, rostros):
        """Vectorización multi-escala"""
        todos_tensors = []
        
        for img in rostros:
            escalas = [112, 128]
            
            for escala in escalas:
                img_resized = cv2.resize(img, (escala, escala))
                if escala > 112:
                    start = (escala - 112) // 2
                    img_resized = img_resized[start:start+112, start:start+112]
                
                for angulo in [0, -5, 5, -10, 10]:
                    if angulo != 0:
                        M = cv2.getRotationMatrix2D((56, 56), angulo, 1.0)
                        img_rot = cv2.warpAffine(img_resized, M, (112, 112))
                    else:
                        img_rot = img_resized
                    
                    img_rgb = cv2.cvtColor(img_rot, cv2.COLOR_BGR2RGB)
                    img_norm = (img_rgb.astype(np.float32) - 127.5) / 128.0
                    img_t = img_norm.transpose(2, 0, 1)
                    todos_tensors.append(img_t)
        
        input_t = torch.from_numpy(np.array(todos_tensors)).to(self.device).half()
        embs = self.model(input_t)
        return F.normalize(embs, p=2, dim=1).cpu().numpy()

    def grabar_video_registro(self, nombre, duracion=30):
        """Graba video automáticamente para registro"""
        print(f"\n🎥 GRABANDO VIDEO PARA REGISTRO")
        print(f"👤 Persona: {nombre}")
        print(f"⏱️  Duración: {duracion} segundos")
        print(f"\n📌 INSTRUCCIONES:")
        print(f"   • Muévete lentamente de lado a lado")
        print(f"   • Gira la cabeza suavemente")
        print(f"   • Prueba diferentes distancias de la cámara")
        print(f"   • La grabación inicia en 3 segundos...\n")
        
        time.sleep(3)
        
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("❌ No se puede abrir la cámara")
            return None
        
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        
        # Crear carpeta de videos si no existe
        os.makedirs('videos_registro', exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        video_path = f'videos_registro/{nombre}_{timestamp}.avi'
        
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        out = cv2.VideoWriter(video_path, fourcc, 20.0, (1280, 720))
        
        inicio = time.time()
        frames_guardados = 0
        
        print("🔴 GRABANDO... (muévete naturalmente)")
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            tiempo_transcurrido = time.time() - inicio
            if tiempo_transcurrido >= duracion:
                break
            
            out.write(frame)
            frames_guardados += 1
            
            # Mostrar progreso cada segundo
            if frames_guardados % 20 == 0:
                tiempo_restante = int(duracion - tiempo_transcurrido)
                print(f"  ⏳ {tiempo_restante}s restantes... ({frames_guardados} frames)")
        
        cap.release()
        out.release()
        
        print(f"\n✅ Video guardado: {video_path}")
        print(f"📊 Total frames: {frames_guardados}\n")
        
        return video_path

    def registrar_desde_video(self, nombre, video_path):
        """Registra una persona desde un video"""
        print(f"\n⚙️  PROCESANDO REGISTRO: {nombre}")
        print(f"📹 Analizando video: {video_path}")
        
        cap = cv2.VideoCapture(video_path)
        
        temp_faces = []
        frame_count = 0
        frames_procesados = 0
        
        while len(temp_faces) < self.fotos_registro:
            ret, frame = cap.read()
            if not ret:
                # Reiniciar video si no hay suficientes frames
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                frames_procesados += 1
                if frames_procesados > 10:  # Máximo 10 pasadas
                    print(f"⚠️  Solo se capturaron {len(temp_faces)} rostros del video")
                    break
                continue
            
            frame_count += 1
            
            # Procesar cada 2 frames
            if frame_count % 2 != 0:
                continue
            
            frame_norm = self.normalizar_iluminacion(frame)
            h, w = frame_norm.shape[:2]
            
            blob = cv2.dnn.blobFromImage(cv2.resize(frame_norm, (300, 300)), 1.0, (300, 300), (104, 177, 123))
            self.net.setInput(blob)
            detections = self.net.forward()
            
            for i in range(detections.shape[2]):
                confidence = detections[0, 0, i, 2]
                if confidence > 0.7:
                    box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                    x, y, x2, y2 = box.astype(int)
                    
                    margen_x = int((x2 - x) * 0.2)
                    margen_y = int((y2 - y) * 0.2)
                    x = max(0, x - margen_x)
                    y = max(0, y - margen_y)
                    x2 = min(w, x2 + margen_x)
                    y2 = min(h, y2 + margen_y)
                    
                    roi = frame_norm[y:y2, x:x2]
                    if roi.size > 0:
                        temp_faces.append(roi.copy())
                        
                        if len(temp_faces) % 100 == 0:
                            print(f"  📸 Capturadas: {len(temp_faces)}/{self.fotos_registro}")
                        
                        if len(temp_faces) >= self.fotos_registro:
                            break
        
        cap.release()
        
        if len(temp_faces) < 100:
            print(f"❌ Error: Solo se detectaron {len(temp_faces)} rostros. Se necesitan al menos 100.")
            return False
        
        # Vectorizar
        print(f"\n🔥 VECTORIZACIÓN NUCLEAR INICIADA")
        print(f"📊 {len(temp_faces)} rostros × {self.vectores_por_foto} variaciones = {len(temp_faces) * self.vectores_por_foto:,} vectores")
        
        all_embs = []
        
        for i in range(0, len(temp_faces), self.batch_size // self.vectores_por_foto):
            lote_base = temp_faces[i:i + self.batch_size // self.vectores_por_foto]
            lote_variado = []
            
            for img in lote_base:
                variaciones = self.augmentar_imagen(img)
                lote_variado.extend(variaciones)
            
            embs = self.procesar_lote_hardcore(lote_variado)
            all_embs.extend(embs)
            
            progreso = min(i + self.batch_size // self.vectores_por_foto, len(temp_faces))
            porcentaje = (progreso / len(temp_faces)) * 100
            print(f"  ⚡ Progreso: {progreso}/{len(temp_faces)} ({porcentaje:.1f}%)")
        
        # Guardar
        db = {}
        if Path(self.db_path).exists():
            with open(self.db_path, 'rb') as f: 
                db = pickle.load(f)
        
        db[nombre] = all_embs
        with open(self.db_path, 'wb') as f:
            pickle.dump(db, f)
        
        print(f"\n✅ REGISTRO COMPLETO")
        print(f"🧬 {len(all_embs):,} vectores guardados para {nombre.upper()}")
        
        self.cargar_base_datos()
        return True

    def modo_reconocimiento_continuo(self):
        """Reconocimiento facial en tiempo real mostrando resultados en terminal"""
        print(f"\n🎥 MODO RECONOCIMIENTO CONTINUO")
        print(f"⌨️  Presiona Ctrl+C para detener\n")
        
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("❌ No se puede abrir la cámara")
            return
        
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        
        frame_count = 0
        ultimo_reporte = time.time()
        detecciones_recientes = []
        
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                frame_count += 1
                
                # Procesar cada 5 frames para performance
                if frame_count % 5 != 0:
                    continue
                
                frame_norm = self.normalizar_iluminacion(frame)
                h, w = frame_norm.shape[:2]
                
                blob = cv2.dnn.blobFromImage(cv2.resize(frame_norm, (300, 300)), 1.0, (300, 300), (104, 177, 123))
                self.net.setInput(blob)
                detections_net = self.net.forward()
                
                for i in range(detections_net.shape[2]):
                    confidence = detections_net[0, 0, i, 2]
                    if confidence > 0.7:
                        box = detections_net[0, 0, i, 3:7] * np.array([w, h, w, h])
                        x, y, x2, y2 = box.astype(int)
                        
                        margen_x = int((x2 - x) * 0.2)
                        margen_y = int((y2 - y) * 0.2)
                        x = max(0, x - margen_x)
                        y = max(0, y - margen_y)
                        x2 = min(w, x2 + margen_x)
                        y2 = min(h, y2 + margen_y)
                        
                        roi = frame_norm[y:y2, x:x2]
                        if roi.size == 0:
                            continue
                        
                        if self.db_embeddings is not None:
                            variaciones = self.augmentar_imagen(roi)[:3]
                            face_vecs = self.procesar_lote_hardcore(variaciones)
                            
                            face_v_mean = np.mean(face_vecs, axis=0, keepdims=True)
                            face_v = torch.tensor(face_v_mean).to(self.device).half()
                            
                            sims = torch.mm(face_v, self.db_embeddings.t())
                            top_k = min(50, sims.shape[1])
                            top_vals, top_idx = torch.topk(sims, top_k, dim=1)
                            
                            nombres_top = [self.db_nombres[idx] for idx in top_idx[0].cpu().numpy()]
                            nombre_ganador = max(set(nombres_top), key=nombres_top.count)
                            score = top_vals[0, 0].item()
                            
                            if score > self.threshold and nombres_top.count(nombre_ganador) > top_k * 0.4:
                                detecciones_recientes.append({
                                    'nombre': nombre_ganador,
                                    'score': score,
                                    'timestamp': time.time()
                                })
                
                # Reportar cada 2 segundos
                if time.time() - ultimo_reporte >= 2:
                    os.system('clear' if os.name != 'nt' else 'cls')
                    print(f"\n{'='*60}")
                    print(f"🎥 RECONOCIMIENTO ACTIVO - Frame {frame_count}")
                    print(f"{'='*60}\n")
                    
                    if detecciones_recientes:
                        # Agrupar por nombre
                        nombres_unicos = {}
                        for det in detecciones_recientes:
                            nombre = det['nombre']
                            if nombre not in nombres_unicos:
                                nombres_unicos[nombre] = []
                            nombres_unicos[nombre].append(det['score'])
                        
                        for nombre, scores in nombres_unicos.items():
                            avg_score = np.mean(scores)
                            print(f"  ✅ {nombre.upper():<20} | Confianza: {avg_score:.1%} | Detecciones: {len(scores)}")
                        
                        detecciones_recientes = []
                    else:
                        print(f"  ⏳ Sin detecciones recientes...")
                    
                    print(f"\n{'='*60}")
                    print(f"⌨️  Presiona Ctrl+C para volver al menú")
                    print(f"{'='*60}\n")
                    
                    ultimo_reporte = time.time()
        
        except KeyboardInterrupt:
            print("\n\n🛑 Deteniendo reconocimiento...\n")
        
        cap.release()

def menu():
    print(f"\n{'='*60}")
    print(f"{'🔥 GPU DESTROYER V4 - AUTO REGISTRO 🔥':^60}")
    print(f"{'='*60}")
    print(f"\n  1. 🎬 Registrar nueva persona (graba automáticamente)")
    print(f"  2. 🎥 Reconocimiento continuo (terminal)")
    print(f"  3. 📊 Ver base de datos")
    print(f"  4. 🗑️  Borrar base de datos")
    print(f"  5. 🚪 Salir")
    print(f"\n{'='*60}\n")

if __name__ == "__main__":
    reconocedor = ReconocedorDestroyeGPU()
    
    while True:
        menu()
        opcion = input("Selecciona opción: ").strip()
        
        if opcion == "1":
            print()
            nombre = input("👤 Nombre de la persona: ").strip().lower()
            if not nombre:
                print("❌ Nombre inválido")
                continue
            
            duracion = input("⏱️  Duración del video en segundos [30]: ").strip()
            duracion = int(duracion) if duracion.isdigit() else 30
            
            # Grabar video
            video_path = reconocedor.grabar_video_registro(nombre, duracion)
            
            if video_path and os.path.exists(video_path):
                # Registrar desde el video
                reconocedor.registrar_desde_video(nombre, video_path)
            else:
                print("❌ Error en la grabación")
        
        elif opcion == "2":
            if reconocedor.db_embeddings is None:
                print("\n❌ Primero debes registrar al menos una persona\n")
            else:
                reconocedor.modo_reconocimiento_continuo()
        
        elif opcion == "3":
            if reconocedor.db_embeddings is not None:
                print(f"\n{'='*60}")
                print(f"📊 PERSONAS REGISTRADAS")
                print(f"{'='*60}\n")
                for nombre in set(reconocedor.db_nombres):
                    count = reconocedor.db_nombres.count(nombre)
                    print(f"  • {nombre.upper():<20} → {count:,} vectores")
                print()
            else:
                print("\n❌ Base de datos vacía\n")
        
        elif opcion == "4":
            confirmar = input("\n⚠️  ¿Seguro que quieres borrar toda la DB? (si/no): ").strip().lower()
            if confirmar == 'si':
                if os.path.exists(reconocedor.db_path):
                    os.remove(reconocedor.db_path)
                reconocedor.db_embeddings = None
                reconocedor.db_nombres = []
                print("🗑️  Base de datos eliminada\n")
            else:
                print("❌ Cancelado\n")
        
        elif opcion == "5":
            print("\n👋 Hasta luego!\n")
            break
        
        else:
            print("\n❌ Opción inválida\n")
