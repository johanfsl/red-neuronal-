import cv2
import torch
import torch.nn.functional as F
import numpy as np
import pickle
import imagezmq
from pathlib import Path
from model import common
from collections import deque
from sklearn.metrics.pairwise import cosine_similarity

# ======================================================
class ServidorReconocimiento:
    def __init__(self, model_path="weights/adaface_ir101_ms1mv3.ckpt"):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"[SERVER] Usando {self.device}")

        # Parámetros ULTRA PRECISOS (aprovechando GPU potente)
        self.db_path = "identidades.pkl"
        self.umbral_base = 0.68  # Más estricto
        self.umbral_dinamico = {}  # Umbral por persona
        self.margen_ambiguedad = 0.15  # Más estricto
        self.muestras_registro = 200  # MÁS muestras = mejor calidad
        self.min_blur_score = 70  # Balance
        self.min_roi_size = 70
        
        # Multi-ventana temporal (varias escalas de tiempo)
        self.historial_corto = deque(maxlen=3)   # Detección rápida
        self.historial_largo = deque(maxlen=10)  # Confirmación robusta
        self.min_consecutivos = 4  # Más frames para confirmar
        
        # Buffer de embeddings recientes para análisis
        self.embeddings_recientes = deque(maxlen=20)
        
        # Calidad adaptativa
        self.calidad_frame_anterior = 0
        
        self.nombre_registro = ""
        self.buffer_fotos = []
        
        # Stats
        self.stats = {
            "total": 0,
            "reconocidos": 0,
            "desconocidos": 0,
            "rechazados": 0
        }

        # Modelo
        print("[MODEL] Cargando AdaFace IR101 (optimizado GPU)...")
        self.modelo = common.iresnet101(num_features=512).to(self.device).eval().half()
        ckpt = torch.load(model_path, map_location=self.device, weights_only=True)
        state = ckpt.get("state_dict", ckpt)
        self.modelo.load_state_dict(
            {k.replace("model.", ""): v for k, v in state.items()},
            strict=False
        )
        
        # Compilar modelo para mayor velocidad (PyTorch 2.0+)
        try:
            self.modelo = torch.compile(self.modelo, mode="max-autotune")
            print("[MODEL] Modelo compilado con torch.compile")
        except:
            print("[MODEL] torch.compile no disponible, usando modelo normal")
        
        print("[MODEL] Listo")

        # Detector
        self.detector = cv2.dnn.readNetFromCaffe(
            "deploy.prototxt",
            "res10_300x300_ssd_iter_140000.caffemodel"
        )
        
        # Usar GPU para detector también
        self.detector.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
        self.detector.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)
        print("[DETECTOR] Usando aceleración CUDA")

        self.db_embeddings = None
        self.db_nombres = []
        self.db_metadata = {}  # Metadata por persona
        self.cargar_db()

    # ==================================================
    def cargar_db(self):
        if not Path(self.db_path).exists():
            print("[DB] Vacía")
            return

        with open(self.db_path, "rb") as f:
            data = pickle.load(f)

        vectores, nombres, metadata = [], [], {}
        db_corregida = False
        
        for nombre, contenido in data.items():
            # Soportar formato viejo y nuevo
            if isinstance(contenido, dict):
                embedding = contenido['embedding']
                metadata[nombre] = contenido.get('metadata', {})
            else:
                embedding = contenido
                metadata[nombre] = {}
            
            if not isinstance(embedding, np.ndarray):
                embedding = np.array(embedding)
            
            if embedding.ndim == 2:
                print(f"[DB] Convirtiendo {nombre} a formato optimizado")
                embedding = np.mean(embedding, axis=0)
                embedding = embedding / np.linalg.norm(embedding)
                data[nombre] = {'embedding': embedding, 'metadata': metadata[nombre]}
                db_corregida = True
            
            if embedding.ndim > 1:
                embedding = embedding.flatten()
            
            if embedding.shape[0] != 512:
                print(f"[ERROR] Embedding de {nombre} incorrecto: {embedding.shape}")
                continue
            
            vectores.append(embedding)
            nombres.append(nombre)

        if db_corregida:
            print("[DB] Guardando DB optimizada...")
            with open(self.db_path, "wb") as f:
                pickle.dump(data, f)

        if vectores:
            vectores_array = np.array(vectores)
            
            self.db_embeddings = torch.tensor(
                vectores_array,
                device=self.device
            ).half()
            self.db_embeddings = F.normalize(self.db_embeddings, p=2, dim=1)
            self.db_nombres = nombres
            self.db_metadata = metadata
            
            # Calcular umbrales adaptativos por persona
            self.calcular_umbrales_adaptativos()
            
            print(f"[DB] ✓ {len(nombres)} identidades cargadas")
            for n in nombres:
                umbral = self.umbral_dinamico.get(n, self.umbral_base)
                print(f"     • {n} (umbral: {umbral:.3f})")

    # ==================================================
    def calcular_umbrales_adaptativos(self):
        """Calcula umbral óptimo por persona basado en similitud intra-clase"""
        if self.db_embeddings is None or len(self.db_nombres) < 2:
            return
        
        # Calcular matriz de similitud
        sims = torch.mm(self.db_embeddings, self.db_embeddings.t()).cpu().numpy()
        
        for i, nombre in enumerate(self.db_nombres):
            # Similitudes con otras personas (negativos)
            mask = np.ones(len(self.db_nombres), dtype=bool)
            mask[i] = False
            max_sim_otros = np.max(sims[i][mask]) if np.any(mask) else 0.0
            
            # Umbral adaptativo: entre mejor match negativo y 1.0
            umbral = min(self.umbral_base, max_sim_otros + 0.15)
            self.umbral_dinamico[nombre] = max(umbral, 0.60)  # Mínimo 0.60

    # ==================================================
    def borrar_db(self):
        if Path(self.db_path).exists():
            Path(self.db_path).unlink()
        
        self.db_embeddings = None
        self.db_nombres = []
        self.db_metadata = {}
        self.umbral_dinamico = {}
        self.historial_corto.clear()
        self.historial_largo.clear()
        self.embeddings_recientes.clear()
        print("[DB] Sistema reiniciado")

    # ==================================================
    def borrar_persona(self, nombre):
        if not Path(self.db_path).exists():
            return False

        with open(self.db_path, "rb") as f:
            data = pickle.load(f)

        if nombre in data:
            del data[nombre]
            with open(self.db_path, "wb") as f:
                pickle.dump(data, f)
            self.cargar_db()
            return True
        return False

    # ==================================================
    def listar_personas(self):
        if not Path(self.db_path).exists():
            return []
        with open(self.db_path, "rb") as f:
            data = pickle.load(f)
        return list(data.keys())

    # ==================================================
    def calcular_calidad_rostro(self, roi):
        """Análisis de calidad multi-métrica"""
        if roi is None or roi.size == 0:
            return False, 0, 0, 0
        
        h, w = roi.shape[:2]
        if h < self.min_roi_size or w < self.min_roi_size:
            return False, 0, 0, 0
        
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        
        # 1. Blur score (varianza Laplaciano)
        blur = cv2.Laplacian(gray, cv2.CV_64F).var()
        
        # 2. Brillo promedio
        brillo = np.mean(gray)
        
        # 3. Contraste (desviación estándar)
        contraste = np.std(gray)
        
        # Validación
        valido = (
            blur >= self.min_blur_score and
            20 < brillo < 235 and
            contraste > 15  # Mínimo contraste
        )
        
        return valido, blur, brillo, contraste

    # ==================================================
    def normalizar_rostro_ultra(self, roi):
        """Normalización ULTRA PRECISA con múltiples capas"""
        if roi is None or roi.size == 0:
            return None

        # 1. Conversión a RGB
        img = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)
        
        # 2. Reducción de ruido adaptativa
        h, w = img.shape[:2]
        if min(h, w) > 150:
            img = cv2.fastNlMeansDenoisingColored(img, None, 8, 8, 7, 21)
        
        # 3. Filtro bilateral (preserva bordes)
        img = cv2.bilateralFilter(img, 5, 60, 60)
        
        # 4. Resize de alta calidad
        img = cv2.resize(img, (112, 112), interpolation=cv2.INTER_LANCZOS4)
        
        # 5. Corrección de iluminación adaptativa (CLAHE en LAB)
        lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
        l = clahe.apply(l)
        img = cv2.cvtColor(cv2.merge([l, a, b]), cv2.COLOR_LAB2RGB)
        
        # 6. Gamma adaptativo FINO
        mean = np.mean(img)
        if mean < 50:
            gamma = 0.45
        elif mean < 70:
            gamma = 0.65
        elif mean < 90:
            gamma = 0.80
        elif mean > 190:
            gamma = 1.6
        elif mean > 170:
            gamma = 1.4
        elif mean > 150:
            gamma = 1.2
        else:
            gamma = 1.0
        
        if gamma != 1.0:
            inv_gamma = 1.0 / gamma
            table = np.array([
                ((i / 255.0) ** inv_gamma) * 255
                for i in np.arange(0, 256)
            ]).astype("uint8")
            img = cv2.LUT(img, table)
        
        # 7. Sharpening suave
        kernel = np.array([[-0.5, -0.5, -0.5],
                          [-0.5,  5.0, -0.5],
                          [-0.5, -0.5, -0.5]])
        img = cv2.filter2D(img, -1, kernel * 0.3)
        img = np.clip(img, 0, 255).astype(np.uint8)
        
        # 8. Normalización ArcFace
        img = img.astype(np.float32)
        img = (img - 127.5) / 128.0
        
        return img.transpose(2, 0, 1)

    # ==================================================
    @torch.no_grad()
    def identificar_ultra(self, roi):
        """Identificación ULTRA PRECISA con múltiples validaciones"""
        if self.db_embeddings is None:
            return "SIN_DB", 0.0, ""
        
        self.stats["total"] += 1

        # Normalizar
        pre = self.normalizar_rostro_ultra(roi)
        if pre is None:
            return "ERROR", 0.0, ""

        # Generar embedding
        t = torch.from_numpy(pre).unsqueeze(0).to(self.device).half()
        emb = F.normalize(self.modelo(t), p=2, dim=1)
        
        # Guardar embedding para análisis temporal
        self.embeddings_recientes.append(emb.cpu().numpy()[0])

        # Calcular similitudes
        sims = torch.mm(emb, self.db_embeddings.t())
        vals, idxs = torch.topk(sims, k=min(len(self.db_nombres), 3), dim=1)

        score_top1 = vals[0][0].item()
        nombre_top1 = self.db_nombres[idxs[0][0].item()]
        
        # Obtener umbral adaptativo
        umbral = self.umbral_dinamico.get(nombre_top1, self.umbral_base)

        # Análisis de ambigüedad (top-2)
        if vals.shape[1] > 1:
            score_top2 = vals[0][1].item()
            nombre_top2 = self.db_nombres[idxs[0][1].item()]
            
            # Si hay dos personas MUY cercanas → rechazar
            if nombre_top1 != nombre_top2 and (score_top1 - score_top2) < self.margen_ambiguedad:
                self.stats["desconocidos"] += 1
                return "DESCONOCIDO", score_top1, f"Ambiguo: {nombre_top1} vs {nombre_top2}"

        # Historial multi-escala
        self.historial_corto.append((nombre_top1, score_top1))
        self.historial_largo.append((nombre_top1, score_top1))
        
        # Análisis de consistencia temporal
        if len(self.historial_largo) >= self.min_consecutivos:
            # Conteo en ventana larga
            nombres_largo = [n for n, s in self.historial_largo]
            conteo = nombres_largo.count(nombre_top1)
            
            # Score promedio ponderado (más peso a recientes)
            scores = [s for n, s in self.historial_largo if n == nombre_top1]
            if scores:
                weights = np.linspace(0.5, 1.0, len(scores))
                score_promedio = np.average(scores, weights=weights)
            else:
                score_promedio = score_top1
            
            # Consistencia en ventana corta (últimos 3 frames)
            if len(self.historial_corto) >= 3:
                nombres_corto = [n for n, s in self.historial_corto]
                consistencia = nombres_corto.count(nombre_top1) / len(nombres_corto)
            else:
                consistencia = 0
            
            # Decisión final ESTRICTA
            if (conteo >= self.min_consecutivos and 
                score_promedio > umbral and 
                consistencia >= 0.67):  # Al menos 2 de 3
                
                self.stats["reconocidos"] += 1
                return nombre_top1.upper(), score_promedio, f"OK ({conteo}/{len(self.historial_largo)})"
        
        # No cumple criterios
        self.stats["desconocidos"] += 1
        return "DESCONOCIDO", score_top1, f"Score: {score_top1:.3f} | Umbral: {umbral:.3f}"

    # ==================================================
    def guardar_registro_ultra(self):
        """Registro ULTRA PRECISO con limpieza agresiva"""
        print(f"\n[REGISTRO] Procesando '{self.nombre_registro}'")
        print(f"[REGISTRO] Analizando {len(self.buffer_fotos)} capturas...")
        
        embeddings_validos = []
        calidades = []
        rechazados = {"blur": 0, "brillo": 0, "contraste": 0, "otro": 0}

        with torch.no_grad():
            for idx, img in enumerate(self.buffer_fotos):
                valido, blur, brillo, contraste = self.calcular_calidad_rostro(img)
                
                if not valido:
                    if blur < self.min_blur_score:
                        rechazados["blur"] += 1
                    elif brillo <= 20 or brillo >= 235:
                        rechazados["brillo"] += 1
                    elif contraste <= 15:
                        rechazados["contraste"] += 1
                    else:
                        rechazados["otro"] += 1
                    continue
                
                pre = self.normalizar_rostro_ultra(img)
                if pre is not None:
                    t = torch.from_numpy(pre).unsqueeze(0).to(self.device).half()
                    emb = F.normalize(self.modelo(t), p=2, dim=1)
                    embeddings_validos.append(emb.cpu().numpy()[0])
                    calidades.append((blur, brillo, contraste))

        total_rechazados = sum(rechazados.values())
        print(f"[REGISTRO] ✓ Válidos: {len(embeddings_validos)}")
        print(f"[REGISTRO] ✗ Rechazados: {total_rechazados}")
        print(f"            - Blur: {rechazados['blur']}")
        print(f"            - Brillo: {rechazados['brillo']}")
        print(f"            - Contraste: {rechazados['contraste']}")
        print(f"            - Otro: {rechazados['otro']}")

        if len(embeddings_validos) < 40:
            print(f"[ERROR] Muy pocas muestras válidas ({len(embeddings_validos)}/40)")
            self.buffer_fotos = []
            return False

        # Limpieza AGRESIVA de outliers
        embeddings = np.array(embeddings_validos)
        
        # Paso 1: Eliminar outliers por distancia euclidiana
        mean_temp = np.mean(embeddings, axis=0)
        distancias = np.linalg.norm(embeddings - mean_temp, axis=1)
        threshold_dist = np.percentile(distancias, 85)  # Top 85%
        mask_dist = distancias <= threshold_dist
        
        # Paso 2: Eliminar outliers por similitud coseno
        sims_internas = cosine_similarity(embeddings)
        sims_promedio = np.mean(sims_internas, axis=1)
        threshold_sim = np.percentile(sims_promedio, 15)  # Bottom 15% out
        mask_sim = sims_promedio >= threshold_sim
        
        # Combinar máscaras
        mask_final = mask_dist & mask_sim
        embeddings_ultra_limpios = embeddings[mask_final]
        
        print(f"[REGISTRO] Limpieza outliers:")
        print(f"            - Distancia: {np.sum(~mask_dist)} eliminados")
        print(f"            - Similitud: {np.sum(~mask_sim)} eliminados")
        print(f"            - Final: {len(embeddings_ultra_limpios)} embeddings")
        
        if len(embeddings_ultra_limpios) < 20:
            print(f"[ERROR] Muy pocos después de limpieza ({len(embeddings_ultra_limpios)})")
            self.buffer_fotos = []
            return False
        
        # Embedding promedio ROBUSTO
        mean_embedding = np.mean(embeddings_ultra_limpios, axis=0)
        mean_embedding = mean_embedding / np.linalg.norm(mean_embedding)
        
        # Metadata de calidad
        calidades_validas = [calidades[i] for i in range(len(calidades)) if mask_final[i]]
        metadata = {
            'num_samples': len(embeddings_ultra_limpios),
            'avg_blur': np.mean([c[0] for c in calidades_validas]),
            'avg_brightness': np.mean([c[1] for c in calidades_validas]),
            'avg_contrast': np.mean([c[2] for c in calidades_validas]),
        }
        
        # Guardar
        data = {}
        if Path(self.db_path).exists():
            with open(self.db_path, "rb") as f:
                data = pickle.load(f)

        data[self.nombre_registro] = {
            'embedding': mean_embedding,
            'metadata': metadata
        }

        with open(self.db_path, "wb") as f:
            pickle.dump(data, f)

        self.cargar_db()
        self.buffer_fotos = []
        print(f"[REGISTRO] ✓✓✓ COMPLETADO ✓✓✓\n")
        return True

    # ==================================================
    def iniciar(self):
        hub = imagezmq.ImageHub(open_port="tcp://*:5555")
        
        print("\n" + "="*70)
        print("🔥 SERVIDOR ULTRA PRECISO - OPTIMIZADO RTX 5060 Ti 16GB 🔥")
        print("="*70)
        print(f"Device: {self.device}")
        print(f"VRAM disponible: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
        print(f"Umbral base: {self.umbral_base}")
        print(f"Muestras registro: {self.muestras_registro}")
        print(f"Capas procesamiento: 8 (denoising, bilateral, CLAHE, gamma, sharp, etc)")
        print(f"Multi-ventana temporal: Corto (3) + Largo (10)")
        print("="*70)
        print("[SERVER] Esperando cliente...\n")

        frame_count = 0
        ultimo_log = 0

        while True:
            tag, jpg = hub.recv_jpg()
            modo = tag.decode() if isinstance(tag, bytes) else tag
            frame_count += 1

            # Comandos
            if modo == "borrar_db":
                self.borrar_db()
                hub.send_reply(b"DB_BORRADA")
                continue
            
            if modo.startswith("borrar_persona:"):
                nombre = modo.split(":")[1]
                exito = self.borrar_persona(nombre)
                hub.send_reply(b"PERSONA_BORRADA" if exito else b"PERSONA_NO_ENCONTRADA")
                continue
            
            if modo == "listar_personas":
                personas = self.listar_personas()
                respuesta = ",".join(personas) if personas else "VACIO"
                hub.send_reply(respuesta.encode())
                continue

            frame = cv2.imdecode(
                np.frombuffer(jpg, dtype=np.uint8),
                cv2.IMREAD_COLOR
            )
            
            if frame is None:
                hub.send_reply(b"ERROR")
                continue

            h, w = frame.shape[:2]
            blob = cv2.dnn.blobFromImage(frame, 1.0, (300, 300), (104, 177, 123))
            self.detector.setInput(blob)
            dets = self.detector.forward()

            registro_terminado = False

            for i in range(dets.shape[2]):
                if dets[0, 0, i, 2] > 0.65:  # Umbral más alto
                    box = dets[0, 0, i, 3:7] * np.array([w, h, w, h])
                    x, y, x2, y2 = box.astype(int)
                    
                    margen = 20
                    x = max(0, x - margen)
                    y = max(0, y - margen)
                    x2 = min(w, x2 + margen)
                    y2 = min(h, y2 + margen)

                    roi = frame[y:y2, x:x2]
                    
                    valido, blur, brillo, contraste = self.calcular_calidad_rostro(roi)
                    
                    if not valido:
                        self.stats["rechazados"] += 1
                        cv2.rectangle(frame, (x, y), (x2, y2), (80, 80, 80), 2)
                        razon = ""
                        if blur < self.min_blur_score:
                            razon = "BORROSO"
                        elif brillo <= 20 or brillo >= 235:
                            razon = "LUZ MALA"
                        elif contraste <= 15:
                            razon = "BAJO CONTRASTE"
                        cv2.putText(frame, razon, (x, y - 10),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (80, 80, 80), 2)
                        continue

                    if modo.startswith("registro:"):
                        self.nombre_registro = modo.split(":")[1]
                        self.buffer_fotos.append(roi.copy())

                        progreso = int((len(self.buffer_fotos) / self.muestras_registro) * 100)
                        cv2.rectangle(frame, (x, y), (x2, y2), (0, 255, 255), 3)
                        cv2.putText(frame, f"REGISTRO {progreso}%", (x, y - 35),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 2)
                        cv2.putText(frame, f"{len(self.buffer_fotos)}/{self.muestras_registro}", (x, y - 10),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
                        
                        # Info de calidad
                        cv2.putText(frame, f"B:{blur:.0f} L:{brillo:.0f} C:{contraste:.0f}",
                                    (x, y2 + 20),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)

                        if len(self.buffer_fotos) >= self.muestras_registro:
                            if self.guardar_registro_ultra():
                                registro_terminado = True
                    else:
                        nombre, score, info = self.identificar_ultra(roi)
                        
                        if nombre in ["DESCONOCIDO", "SIN_DB", "ERROR"]:
                            color = (0, 0, 255)
                        else:
                            color = (0, 255, 0)

                        cv2.rectangle(frame, (x, y), (x2, y2), color, 3)
                        
                        texto = f"{nombre} {score:.3f}"
                        (tw, th), _ = cv2.getTextSize(texto, cv2.FONT_HERSHEY_SIMPLEX, 0.9, 2)
                        cv2.rectangle(frame, (x, y - th - 20), (x + tw + 10, y), color, -1)
                        cv2.putText(frame, texto, (x + 5, y - 10),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
                        
                        # Info adicional
                        if info:
                            cv2.putText(frame, info, (x, y2 + 20),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

            if registro_terminado:
                hub.send_reply(b"DONE")
            else:
                _, out = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), 92])
                hub.send_reply(out.tobytes())

            # Stats cada 100 frames
            if frame_count - ultimo_log >= 100:
                total = self.stats["total"]
                if total > 0:
                    print(f"\n[STATS] Frames: {frame_count}")
                    print(f"        Total detecciones: {total}")
                    print(f"        Reconocidos: {self.stats['reconocidos']} ({self.stats['reconocidos']/total*100:.1f}%)")
                    print(f"        Desconocidos: {self.stats['desconocidos']} ({self.stats['desconocidos']/total*100:.1f}%)")
                    print(f"        Rechazados: {self.stats['rechazados']}\n")
                ultimo_log = frame_count


# ======================================================
if __name__ == "__main__":
    servidor = ServidorReconocimiento()
    servidor.iniciar()
