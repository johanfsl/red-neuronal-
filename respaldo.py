import cv2
import torch
import torch.nn.functional as F
import numpy as np
import pickle
import os
import time
from pathlib import Path
from collections import defaultdict, deque
from model import common

class ReconocedorUltra:
    def __init__(self, model_path='weights/adaface_ir101_ms1mv3.ckpt'):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.db_path = 'identidades_v6.pkl'
        
        # Parametros de seguridad extrema
        self.umbral_estricto = 0.83       
        self.margen_seguridad = 0.12     # Diferencia minima entre el 1ro y el 2do lugar
        self.muestras_registro = 120     
        
        self.model = common.iresnet101(num_features=512).to(self.device).eval().half()
        try:
            ckpt = torch.load(model_path, map_location=self.device, weights_only=True)
            state_dict = ckpt.get('state_dict', ckpt)
            self.model.load_state_dict({k.replace('model.', ''): v for k, v in state_dict.items()}, strict=False)
        except Exception as e:
            print(f"Error critico: {e}")

        self.net = cv2.dnn.readNetFromCaffe("deploy.prototxt", "res10_300x300_ssd_iter_140000.caffemodel")
        
        self.db_nombres = []
        self.db_embeddings = None
        self.cargar_db()

        self.modo_registro = False
        self.nombre_registro = ""
        self.fotos_buffer = []

    def cargar_db(self):
        if Path(self.db_path).exists():
            with open(self.db_path, 'rb') as f:
                data = pickle.load(f)
            vectores, nombres = [], []
            for n, lista in data.items():
                vectores.extend(lista)
                nombres.extend([n] * len(lista))
            if vectores:
                self.db_embeddings = torch.tensor(np.array(vectores)).to(self.device).half()
                self.db_embeddings = F.normalize(self.db_embeddings, p=2, dim=1)
                self.db_nombres = nombres

    def normalizar_rostro(self, roi):
        if roi.size == 0: return None
        # Balanceo de blancos y sombras agresivo
        roi_bw = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        roi_norm = cv2.equalizeHist(roi_bw)
        roi_final = cv2.cvtColor(roi_norm, cv2.COLOR_GRAY2BGR)
        
        roi_final = cv2.resize(roi_final, (112, 112))
        img = (roi_final.astype(np.float32) - 127.5) / 128.0
        return img.transpose(2, 0, 1)

    @torch.no_grad()
    def identificar(self, roi):
        if self.db_embeddings is None or len(self.db_nombres) == 0:
            return "SISTEMA VACIO", 0.0
        
        pre = self.normalizar_rostro(roi)
        if pre is None: return "ERROR", 0.0
        
        t = torch.from_numpy(pre).unsqueeze(0).to(self.device).half()
        emb = self.model(t)
        emb = F.normalize(emb, p=2, dim=1)
        
        sims = torch.mm(emb, self.db_embeddings.t())
        
        # Obtenemos los dos mejores candidatos
        vals, idxs = torch.topk(sims, k=min(2, sims.shape[1]), dim=1)
        
        conf_1 = vals[0][0].item()
        nombre_1 = self.db_nombres[idxs[0][0].item()]
        
        # Logica anti-Johan (Si hay dos personas muy parecidas, dudar)
        if vals.shape[1] > 1:
            conf_2 = vals[0][1].item()
            if (conf_1 - conf_2) < self.margen_seguridad:
                return "INSEGURO", conf_1

        if conf_1 > self.umbral_estricto:
            return nombre_1.upper(), conf_1
        
        return "DESCONOCIDO", conf_1

    def guardar_registro(self):
        print(f"Calculando identidad robusta para {self.nombre_registro}...")
        vectores = []
        with torch.no_grad():
            for img in self.fotos_buffer:
                pre = self.normalizar_rostro(img)
                if pre is not None:
                    t = torch.from_numpy(pre).unsqueeze(0).to(self.device).half()
                    emb = F.normalize(self.model(t), p=2, dim=1).detach().cpu().numpy()
                    vectores.append(emb[0])
        
        db_data = {}
        if Path(self.db_path).exists():
            with open(self.db_path, 'rb') as f: db_data = pickle.load(f)
        
        db_data[self.nombre_registro] = vectores
        with open(self.db_path, 'wb') as f: pickle.dump(db_data, f)
        
        self.cargar_db()
        self.modo_registro = False
        self.fotos_buffer = []
        print("Usuario registrado con exito.")

    def run(self):
        cap = cv2.VideoCapture(0)
        while True:
            ret, frame = cap.read()
            if not ret: break
            
            h, w = frame.shape[:2]
            blob = cv2.dnn.blobFromImage(frame, 1.0, (300, 300), (104, 177, 123))
            self.net.setInput(blob)
            detecciones = self.net.forward()
            
            for i in range(detecciones.shape[2]):
                conf_det = detecciones[0, 0, i, 2]
                if conf_det > 0.5:
                    box = detecciones[0, 0, i, 3:7] * np.array([w, h, w, h])
                    x, y, x2, y2 = box.astype("int")
                    x, y, x2, y2 = max(0, x), max(0, y), min(w, x2), min(h, y2)
                    
                    roi = frame[y:y2, x:x2]
                    if roi.size == 0: continue

                    if self.modo_registro:
                        # Forzamos captura solo si hay movimiento/gestos
                        self.fotos_buffer.append(roi.copy())
                        progreso = int((len(self.fotos_buffer) / self.muestras_registro) * 100)
                        
                        msg = "MUEVE LA CARA / HABLA" if progreso < 80 else "LISTO, QUEDATE QUIETO"
                        cv2.putText(frame, f"{msg}: {progreso}%", (x, y-10), 1, 1.2, (0, 255, 255), 2)
                        
                        if len(self.fotos_buffer) >= self.muestras_registro:
                            self.guardar_registro()
                    else:
                        nombre, score = self.identificar(roi)
                        color = (0, 255, 0) if nombre not in ["DESCONOCIDO", "INSEGURO"] else (0, 0, 255)
                        cv2.rectangle(frame, (x, y), (x2, y2), color, 2)
                        cv2.putText(frame, f"{nombre} {score:.2f}", (x, y-10), 1, 1.2, color, 2)

            cv2.imshow("Control de Acceso Pro", frame)
            
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'): break
            elif key == ord('r') and not self.modo_registro:
                nombre = input("Nombre para registro: ").strip().lower()
                if nombre:
                    self.nombre_registro = nombre
                    self.modo_registro = True
                    self.fotos_buffer = []

        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    ReconocedorUltra().run()
