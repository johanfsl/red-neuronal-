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
import time
import json
from collections import defaultdict, deque

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

class ReconocedorCompleto:
    def __init__(self, model_path='weights/adaface_ir101_ms1mv3.ckpt'):
        descargar_modelos()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.db_path = 'identidades_nuclear.pkl'
        self.log_path = 'recognition_log.json'
        
        # --- PARÁMETROS ---
        self.threshold = 0.68
        self.threshold_estricto = 0.75
        self.fotos_registro = 1500
        self.batch_size = 256
        self.vectores_por_foto = 8
        
        print(f"🔥 GPU DESTROYER V5 - SISTEMA COMPLETO")
        print(f"🖥️  Dispositivo: {self.device}")
        
        # Cargar AdaFace
        self.model = common.iresnet101(num_features=512).to(self.device).eval().half()
        try:
            checkpoint = torch.load(model_path, map_location=self.device, weights_only=True)
            state_dict = checkpoint.get('state_dict', checkpoint)
            self.model.load_state_dict({k.replace('model.', ''): v for k, v in state_dict.items()}, strict=False)
            print("✅ AdaFace IR101 cargado")
        except Exception as e:
            print(f"❌ Error: {e}")

        # Detector SSD
        self.net = cv2.dnn.readNetFromCaffe(PROTOTXT, MODEL_FILE)
        self.net.setPreferableBackend(cv2.dnn.DNN_BACKEND_DEFAULT)
        self.net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
        
        self.db_nombres = []
        self.db_embeddings = None 
        self.cargar_base_datos()

        # --- ESTADOS ---
        self.modo_registro = False
        self.nombre_nuevo = ""
        self.temp_faces = []
        self.inicio_registro = None
        self.duracion_registro = 30
        
        # --- ESTADÍSTICAS ---
        self.historial_detecciones = defaultdict(list)
        self.confianzas_recientes = defaultdict(lambda: deque(maxlen=30))
        self.fps_counter = deque(maxlen=30)
        self.ultimo_tiempo = time.time()
        
        # --- MODOS ---
        self.modo_seguro = False
        self.mostrar_stats = True
        self.grabar_video = False
        self.video_writer = None
        self.anti_spoofing = True
        self.personas_autorizadas = set()
        
        # Directorios
        os.makedirs('logs', exist_ok=True)
        os.makedirs('videos_capturados', exist_ok=True)
        os.makedirs('capturas', exist_ok=True)

    def normalizar_iluminacion(self, img):
        lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
        l = clahe.apply(l)
        lab = cv2.merge([l, a, b])
        return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)

    def augmentar_imagen(self, img):
        variaciones = [img]
        variaciones.append(cv2.convertScaleAbs(img, alpha=1.3, beta=30))
        variaciones.append(cv2.convertScaleAbs(img, alpha=0.7, beta=-30))
        variaciones.append(cv2.GaussianBlur(img, (5, 5), 0))
        kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
        variaciones.append(cv2.filter2D(img, -1, kernel))
        variaciones.append(cv2.convertScaleAbs(img, alpha=1.5, beta=0))
        gamma = 1.5
        invGamma = 1.0 / gamma
        table = np.array([((i / 255.0) ** invGamma) * 255 for i in range(256)]).astype("uint8")
        variaciones.append(cv2.LUT(img, table))
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        hsv[:,:,2] = cv2.add(hsv[:,:,2], 20)
        variaciones.append(cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR))
        return variaciones

    def detectar_spoofing(self, roi):
        if not self.anti_spoofing:
            return True
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
        hist = cv2.calcHist([gray], [0], None, [256], [0, 256])
        hist_std = np.std(hist)
        es_real = laplacian_var > 100 and hist_std > 50
        return es_real

    def cargar_base_datos(self):
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

    def calcular_fps(self):
        tiempo_actual = time.time()
        fps = 1.0 / (tiempo_actual - self.ultimo_tiempo + 1e-6)
        self.ultimo_tiempo = tiempo_actual
        self.fps_counter.append(fps)
        return np.mean(self.fps_counter)

    def dibujar_interfaz_avanzada(self, frame):
        h, w = frame.shape[:2]
        
        # Panel superior
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (w, 150), (0, 0, 0), -1)
        frame = cv2.addWeighted(overlay, 0.75, frame, 0.25, 0)
        
        # Título
        titulo = "GPU DESTROYER V5 - SISTEMA COMPLETO"
        if self.modo_seguro:
            titulo += " [SEGURO]"
        cv2.putText(frame, titulo, (20, 35), cv2.FONT_HERSHEY_DUPLEX, 0.9, (0, 255, 255), 2)
        
        # Controles
        if not self.modo_registro:
            cv2.putText(frame, "[R]Registrar [L]Listar [S]Stats [V]Video [F]Foto [Q]Salir", 
                       (20, 65), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
            cv2.putText(frame, "[M]Seguro [A]AntiSpoof [W]Whitelist [T]Stats [C]BorrarDB", 
                       (20, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
        else:
            tiempo_restante = max(0, self.duracion_registro - (time.time() - self.inicio_registro))
            cv2.putText(frame, f"GRABANDO: {int(tiempo_restante)}s - Muevete naturalmente", 
                       (20, 65), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        
        # Info DB
        if self.db_embeddings is not None:
            personas = len(set(self.db_nombres))
            cv2.putText(frame, f"DB: {personas} personas", (20, 115), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        else:
            cv2.putText(frame, "DB: Vacia", (20, 115), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        
        # Estados
        estados = []
        if self.modo_seguro:
            estados.append(("SEGURO", (0, 165, 255)))
        if self.anti_spoofing:
            estados.append(("ANTI-SPOOF", (0, 255, 0)))
        if self.grabar_video:
            estados.append(("REC", (0, 0, 255)))
        if len(self.personas_autorizadas) > 0:
            estados.append((f"WL:{len(self.personas_autorizadas)}", (255, 165, 0)))
        
        x_estado = 300
        for estado, color in estados:
            cv2.putText(frame, estado, (x_estado, 115), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
            x_estado += 120
        
        # GPU y FPS
        if torch.cuda.is_available():
            gpu_mem = torch.cuda.memory_allocated() / 1024**3
            cv2.putText(frame, f"GPU: {gpu_mem:.2f}GB", (w - 200, 35), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        
        fps = self.calcular_fps()
        cv2.putText(frame, f"FPS: {fps:.1f}", (w - 200, 65), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        # Hora
        hora = datetime.now().strftime("%H:%M:%S")
        cv2.putText(frame, hora, (w - 200, 95), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        
        # Barra progreso
        if self.modo_registro and len(self.temp_faces) > 0:
            progreso = min(len(self.temp_faces), self.fotos_registro)
            porcentaje = (progreso / self.fotos_registro) * 100
            
            barra_w = 500
            barra_h = 40
            barra_x = (w - barra_w) // 2
            barra_y = h - 100
            
            cv2.rectangle(frame, (barra_x, barra_y), (barra_x + barra_w, barra_y + barra_h), (255, 255, 255), 3)
            fill_w = int((barra_w - 6) * (progreso / self.fotos_registro))
            cv2.rectangle(frame, (barra_x + 3, barra_y + 3), 
                         (barra_x + 3 + fill_w, barra_y + barra_h - 3), (0, 255, 0), -1)
            
            texto = f"{progreso}/{self.fotos_registro} ({porcentaje:.1f}%)"
            cv2.putText(frame, texto, (barra_x + barra_w//2 - 120, barra_y - 15), 
                       cv2.FONT_HERSHEY_DUPLEX, 0.8, (255, 255, 255), 2)
        
        # Panel stats
        if self.mostrar_stats and not self.modo_registro:
            self.dibujar_panel_stats(frame)
        
        return frame

    def dibujar_panel_stats(self, frame):
        h, w = frame.shape[:2]
        panel_w = 300
        panel_x = w - panel_w - 10
        panel_y = 160
        
        overlay = frame.copy()
        cv2.rectangle(overlay, (panel_x, panel_y), (w - 10, h - 10), (0, 0, 0), -1)
        frame = cv2.addWeighted(overlay, 0.7, frame, 0.3, 0)
        cv2.rectangle(frame, (panel_x, panel_y), (w - 10, h - 10), (0, 255, 255), 2)
        
        cv2.putText(frame, "ESTADISTICAS", (panel_x + 10, panel_y + 25), 
                   cv2.FONT_HERSHEY_DUPLEX, 0.7, (0, 255, 255), 2)
        
        y_offset = panel_y + 55
        
        if self.historial_detecciones:
            cv2.putText(frame, "Ultima hora:", (panel_x + 10, y_offset), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            y_offset += 25
            
            hora_actual = time.time()
            for nombre in sorted(self.historial_detecciones.keys()):
                timestamps = [t for t in self.historial_detecciones[nombre] if hora_actual - t < 3600]
                if timestamps and y_offset < h - 30:
                    if self.confianzas_recientes[nombre]:
                        confianza_media = np.mean(list(self.confianzas_recientes[nombre]))
                        texto = f"{nombre[:15]}: {len(timestamps)} ({confianza_media:.0%})"
                    else:
                        texto = f"{nombre[:15]}: {len(timestamps)}"
                    cv2.putText(frame, texto, (panel_x + 15, y_offset), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 1)
                    y_offset += 20

    def procesar_reconocimiento(self, frame, frame_norm):
        h, w = frame_norm.shape[:2]
        
        blob = cv2.dnn.blobFromImage(cv2.resize(frame_norm, (300, 300)), 1.0, (300, 300), (104, 177, 123))
        self.net.setInput(blob)
        detections = self.net.forward()
        
        resultados = []
        
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
                if roi.size == 0:
                    continue
                
                if self.modo_registro:
                    if self.inicio_registro is None:
                        self.inicio_registro = time.time()
                    
                    tiempo_transcurrido = time.time() - self.inicio_registro
                    
                    if tiempo_transcurrido <= self.duracion_registro:
                        self.temp_faces.append(roi.copy())
                        color = (0, 255, 255)
                        label = f"CAPTURANDO: {len(self.temp_faces)}"
                        resultados.append((x, y, x2, y2, color, label, confidence, None))
                    elif len(self.temp_faces) >= 100:
                        print(f"\n⚙️ Procesando {len(self.temp_faces)} rostros...")
                        self.guardar_sujeto_nuclear()
                else:
                    if self.db_embeddings is not None:
                        es_real = self.detectar_spoofing(roi)
                        
                        if not es_real:
                            color = (0, 0, 255)
                            label = "POSIBLE FOTO"
                            sublabel = "Spoofing"
                            resultados.append((x, y, x2, y2, color, label, confidence, sublabel))
                            continue
                        
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
                        
                        threshold_actual = self.threshold_estricto if self.modo_seguro else self.threshold
                        
                        if score > threshold_actual and nombres_top.count(nombre_ganador) > top_k * 0.4:
                            self.historial_detecciones[nombre_ganador].append(time.time())
                            self.confianzas_recientes[nombre_ganador].append(score)
                            
                            if len(self.personas_autorizadas) > 0:
                                if nombre_ganador in self.personas_autorizadas:
                                    color = (0, 255, 0)
                                    label = f"✓ {nombre_ganador.upper()}"
                                else:
                                    color = (0, 165, 255)
                                    label = f"⚠ {nombre_ganador.upper()}"
                            else:
                                color = (0, 255, 0)
                                label = f"{nombre_ganador.upper()}"
                            
                            sublabel = f"{score:.1%}"
                        else:
                            color = (0, 165, 255)
                            label = "DESCONOCIDO"
                            sublabel = f"{score:.1%}"
                        
                        resultados.append((x, y, x2, y2, color, label, confidence, sublabel))
                    else:
                        color = (255, 255, 255)
                        label = "SIN DATOS"
                        resultados.append((x, y, x2, y2, color, label, confidence, None))
        
        return resultados

    def dibujar_detecciones(self, frame, detecciones):
        for det in detecciones:
            if len(det) == 8:
                x, y, x2, y2, color, label, confidence, sublabel = det
            else:
                x, y, x2, y2, color, label, confidence = det
                sublabel = None
            
            # Caja con sombra
            cv2.rectangle(frame, (x+3, y+3), (x2+3, y2+3), (0, 0, 0), 3)
            cv2.rectangle(frame, (x, y), (x2, y2), color, 3)
            
            # Esquinas
            corner_len = 20
            cv2.line(frame, (x, y), (x + corner_len, y), color, 4)
            cv2.line(frame, (x, y), (x, y + corner_len), color, 4)
            cv2.line(frame, (x2, y), (x2 - corner_len, y), color, 4)
            cv2.line(frame, (x2, y), (x2, y + corner_len), color, 4)
            
            # Etiqueta
            label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_DUPLEX, 0.8, 2)[0]
            cv2.rectangle(frame, (x, y - 45), (x + label_size[0] + 25, y), color, -1)
            cv2.putText(frame, label, (x + 8, y - 18), 
                       cv2.FONT_HERSHEY_DUPLEX, 0.8, (255, 255, 255), 2)
            
            if sublabel:
                cv2.putText(frame, sublabel, (x + 8, y - 5), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        return frame

    def guardar_sujeto_nuclear(self):
        print(f"🔥 VECTORIZACIÓN: {self.nombre_nuevo}")
        print(f"📊 {len(self.temp_faces)} rostros × {self.vectores_por_foto} variaciones")
        
        all_embs = []
        
        for i in range(0, len(self.temp_faces), self.batch_size // self.vectores_por_foto):
            lote_base = self.temp_faces[i:i + self.batch_size // self.vectores_por_foto]
            lote_variado = []
            
            for img in lote_base:
                variaciones = self.augmentar_imagen(img)
                lote_variado.extend(variaciones)
            
            embs = self.procesar_lote_hardcore(lote_variado)
            all_embs.extend(embs)
            
            progreso = min(i + self.batch_size // self.vectores_por_foto, len(self.temp_faces))
            print(f"  ⚡ {progreso}/{len(self.temp_faces)}")
        
        db = {}
        if Path(self.db_path).exists():
            with open(self.db_path, 'rb') as f: 
                db = pickle.load(f)
        
        db[self.nombre_nuevo] = all_embs
        with open(self.db_path, 'wb') as f:
            pickle.dump(db, f)
        
        print(f"✅ {len(all_embs):,} vectores para {self.nombre_nuevo.upper()}\n")
        
        self.modo_registro = False
        self.temp_faces = []
        self.inicio_registro = None
        self.cargar_base_datos()

    def exportar_estadisticas(self):
        stats = {
            'timestamp': datetime.now().isoformat(),
            'personas_registradas': len(set(self.db_nombres)) if self.db_nombres else 0,
            'total_vectores': len(self.db_nombres) if self.db_nombres else 0,
            'detecciones_ultima_hora': {},
            'modo_seguro': self.modo_seguro,
            'anti_spoofing': self.anti_spoofing
        }
        
        hora_actual = time.time()
        for nombre, timestamps in self.historial_detecciones.items():
            recientes = [t for t in timestamps if hora_actual - t < 3600]
            if recientes:
                stats['detecciones_ultima_hora'][nombre] = len(recientes)
        
        filename = f"logs/stats_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(filename, 'w') as f:
            json.dump(stats, f, indent=2)
        
        print(f"\n📊 Stats: {filename}\n")

    def capturar_frame(self, frame):
        filename = f"capturas/captura_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg"
        cv2.imwrite(filename, frame)
        print(f"\n📸 Captura: {filename}\n")

    def toggle_grabacion(self, frame_shape):
        if not self.grabar_video:
            filename = f"videos_capturados/video_{datetime.now().strftime('%Y%m%d_%H%M%S')}.mp4"
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            self.video_writer = cv2.VideoWriter(filename, fourcc, 20.0, 
                                               (frame_shape[1], frame_shape[0]))
            self.grabar_video = True
            print(f"\n🎥 Grabando: {filename}\n")
        else:
            if self.video_writer:
                self.video_writer.release()
            self.grabar_video = False
            print("\n⏹️  Grabación detenida\n")

    def mostrar_lista_personas(self):
        if self.db_embeddings is not None:
            print(f"\n{'='*70}")
            print(f"📊 PERSONAS REGISTRADAS")
            print(f"{'='*70}")
            
            for nombre in sorted(set(self.db_nombres)):
                count = self.db_nombres.count(nombre)
                hora_actual = time.time()
                detecciones = len([t for t in self.historial_detecciones[nombre] 
                                  if hora_actual - t < 3600])
                
                if self.confianzas_recientes[nombre]:
                    conf_media = np.mean(list(self.confianzas_recientes[nombre]))
                    conf_str = f"{conf_media:.1%}"
                else:
                    conf_str = "N/A"
                
                whitelist = "✓" if nombre in self.personas_autorizadas else ""
                
                print(f"  • {nombre.upper():<20} | Vec: {count:>6,} | "
                      f"Det/h: {detecciones:>3} | Conf: {conf_str} {whitelist}")
            
            print(f"{'='*70}\n")
        else:
            print("\n❌ DB vacía\n")

    def gestionar_whitelist(self):
        print(f"\n{'='*60}")
        print("WHITELIST")
        print(f"{'='*60}")
        
        if not self.db_nombres:
            print("❌ No hay personas\n")
            return
        
        print("\nPersonas:")
        personas = sorted(set(self.db_nombres))
        for i, nombre in enumerate(personas, 1):
            status = "✓" if nombre in self.personas_autorizadas else " "
            print(f"  {i}. [{status}] {nombre.upper()}")
        
        print("\nOpciones: Número | A-todas | N-ninguna | Enter-salir")
        opcion = input("Selección: ").strip()
        
        if opcion.lower() == 'a':
            self.personas_autorizadas = set(personas)
            print("✅ Todas\n")
        elif opcion.lower() == 'n':
            self.personas_autorizadas.clear()
            print("❌ Vacía\n")
        elif opcion.isdigit():
            idx = int(opcion) - 1
            if 0 <= idx < len(personas):
                nombre = personas[idx]
                if nombre in self.personas_autorizadas:
                    self.personas_autorizadas.remove(nombre)
                    print(f"❌ {nombre.upper()} removido\n")
                else:
                    self.personas_autorizadas.add(nombre)
                    print(f"✅ {nombre.upper()} autorizado\n")

    def run(self):
        print("\n🎥 Intentando abrir cámara...")
        cap = cv2.VideoCapture(0)
        
        if not cap.isOpened():
            print("❌ No se puede abrir VideoCapture(0)")
            print("Probando con /dev/video0...")
            cap = cv2.VideoCapture("/dev/video0")
        
        if not cap.isOpened():
            print("❌ Error: No se puede acceder a la cámara")
            print("\nSoluciones:")
            print("  1. ls -la /dev/video*")
            print("  2. sudo usermod -aG video $USER")
            print("  3. Reinicia sesión")
            return
        
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        cap.set(cv2.CAP_PROP_FPS, 30)
        
        print("✅ Cámara abierta")
        print("\n" + "="*70)
        print("🎥 SISTEMA ACTIVO")
        print("="*70)
        print("\n📌 CONTROLES:")
        print("  [R]Registrar [L]Listar [S]Stats [V]Video [F]Foto")
        print("  [M]Seguro [A]AntiSpoof [W]Whitelist [T]Stats [C]Borrar [Q]Salir")
        print("\n" + "="*70 + "\n")
        
        # Crear ventana
        cv2.namedWindow("GPU DESTROYER V5", cv2.WINDOW_NORMAL)
        
        while True:
            ret, frame = cap.read()
            if not ret:
                print("❌ Error leyendo frame")
                break
            
            frame_norm = self.normalizar_iluminacion(frame)
            detecciones = self.procesar_reconocimiento(frame, frame_norm)
            frame = self.dibujar_detecciones(frame, detecciones)
            frame = self.dibujar_interfaz_avanzada(frame)
            
            if self.grabar_video and self.video_writer:
                self.video_writer.write(frame)
            
            cv2.imshow("GPU DESTROYER V5", frame)
            
            key = cv2.waitKey(1) & 0xFF
            
            if key == ord('q'):
                break
            elif key == ord('r') and not self.modo_registro:
                self.nombre_nuevo = input("\n👤 Nombre: ").strip().lower()
                if self.nombre_nuevo:
                    duracion = input(f"⏱️  Duración [30]: ").strip()
                    self.duracion_registro = int(duracion) if duracion.isdigit() else 30
                    self.modo_registro = True
                    self.temp_faces = []
                    self.inicio_registro = None
                    print(f"\n🎬 Registrando: {self.nombre_nuevo.upper()}\n")
            elif key == ord('l'):
                self.mostrar_lista_personas()
            elif key == ord('s'):
                self.exportar_estadisticas()
            elif key == ord('v'):
                self.toggle_grabacion(frame.shape)
            elif key == ord('m'):
                self.modo_seguro = not self.modo_seguro
                print(f"\n🔒 Seguro: {'ON' if self.modo_seguro else 'OFF'}\n")
            elif key == ord('a'):
                self.anti_spoofing = not self.anti_spoofing
                print(f"\n🛡️  AntiSpoof: {'ON' if self.anti_spoofing else 'OFF'}\n")
            elif key == ord('t'):
                self.mostrar_stats = not self.mostrar_stats
            elif key == ord('f'):
                self.capturar_frame(frame)
            elif key == ord('w'):
                self.gestionar_whitelist()
            elif key == ord('c'):
                confirmar = input("\n⚠️  ¿Borrar DB? (si/no): ").strip().lower()
                if confirmar == 'si':
                    if os.path.exists(self.db_path):
                        os.remove(self.db_path)
                    self.db_embeddings = None
                    self.db_nombres = []
                    self.historial_detecciones.clear()
                    self.confianzas_recientes.clear()
                    print("🗑️  Borrada\n")
        
        cap.release()
        if self.video_writer:
            self.video_writer.release()
        cv2.destroyAllWindows()
        print("\n👋 Hasta luego\n")

if __name__ == "__main__":
    print("""
    ╔═══════════════════════════════════════════════════════════╗
    ║     🔥 GPU DESTROYER V5 - SISTEMA COMPLETO 🔥            ║
    ╠═══════════════════════════════════════════════════════════╣
    ║  ✓ Reconocimiento multi-escala + anti-luz                ║
    ║  ✓ Anti-spoofing (detección fotos)                       ║
    ║  ✓ Modo seguro + Whitelist                               ║
    ║  ✓ Estadísticas tiempo real                              ║
    ║  ✓ Grabación video + capturas                            ║
    ║  ✓ ~12,000 vectores/persona                              ║
    ╚═══════════════════════════════════════════════════════════╝
    """)
    
    ReconocedorCompleto().run()
