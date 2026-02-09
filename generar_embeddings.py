import torch
import cv2
import numpy as np
from pathlib import Path
import pickle
from tqdm import tqdm
import sys
import os

# Importar tu código base
ruta_base = os.path.dirname(os.path.abspath(__file__))
sys.path.append(ruta_base)

try:
    from model import common
except ImportError:
    print("❌ ERROR: No se encuentra 'common.py'")
    sys.exit(1)

def generar_embeddings_personas(dataset_path='dataset/personas', 
                                output_path='embeddings_personas.pkl',
                                modelo_path='weights/adaface_ir101_ms1mv3.ckpt'):
    """Genera embeddings para todas las personas en el dataset"""
    
    print("\n" + "="*60)
    print("  GENERADOR DE EMBEDDINGS - AdaFace SOTA")
    print("="*60)
    
    # Configurar dispositivo
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"🖥️  Dispositivo: {device}")
    
    if device.type == 'cuda':
        print(f"🎮 GPU: {torch.cuda.get_device_name(0)}")
        torch.backends.cudnn.benchmark = True
    
    # Cargar modelo
    print("\n📦 Cargando modelo AdaFace...")
    model = common.iresnet101(num_features=512).to(device)
    
    # Cargar pesos
    if not os.path.exists(modelo_path):
        print(f"❌ ERROR: No se encuentra el modelo en {modelo_path}")
        return
    
    checkpoint = torch.load(modelo_path, map_location=device)
    state_dict = checkpoint['state_dict'] if 'state_dict' in checkpoint else checkpoint
    
    # Limpiar llaves
    new_state_dict = {}
    for k, v in state_dict.items():
        name = k[6:] if k.startswith('model.') else k
        new_state_dict[name] = v
    
    model.load_state_dict(new_state_dict, strict=False)
    model.eval()
    
    # Usar FP16 si hay CUDA
    if device.type == 'cuda':
        model = model.half()
        print("✅ Modelo en FP16 (optimizado)")
    
    print("✅ Modelo cargado correctamente\n")
    
    embeddings_dict = {}
    dataset_path = Path(dataset_path)
    
    if not dataset_path.exists():
        print(f"❌ ERROR: No existe la carpeta {dataset_path}")
        print("💡 Primero ejecuta 'capturar_persona.py' para crear el dataset")
        return
    
    # Listar personas disponibles
    personas = [p for p in dataset_path.iterdir() if p.is_dir()]
    
    if not personas:
        print(f"❌ ERROR: No hay personas en {dataset_path}")
        print("💡 Ejecuta 'capturar_persona.py' para agregar personas")
        return
    
    print(f"👥 Personas encontradas: {len(personas)}")
    for p in personas:
        num_imgs = len(list(p.glob('*.jpg'))) + len(list(p.glob('*.png')))
        print(f"   - {p.name}: {num_imgs} imágenes")
    
    print("\n" + "-"*60)
    
    # Procesar cada persona
    for persona_dir in personas:
        nombre = persona_dir.name
        print(f"\n👤 Procesando: {nombre.upper()}")
        
        embeddings_persona = []
        imagenes = list(persona_dir.glob('*.jpg')) + list(persona_dir.glob('*.png'))
        
        if not imagenes:
            print(f"  ⚠️  No hay imágenes para {nombre}")
            continue
        
        for img_path in tqdm(imagenes, desc=f"  Generando embeddings", 
                            bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt}'):
            try:
                # Leer imagen
                img = cv2.imread(str(img_path))
                if img is None:
                    continue
                
                # Preprocesar
                img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                img_resized = cv2.resize(img_rgb, (112, 112))
                
                # Convertir a tensor
                img_tensor = torch.tensor(img_resized).permute(2, 0, 1).unsqueeze(0)
                img_tensor = img_tensor.to(device).float() / 255.0
                
                # FP16 si es CUDA
                if device.type == 'cuda':
                    img_tensor = img_tensor.half()
                
                # Obtener embedding
                with torch.no_grad():
                    embedding = model(img_tensor)
                
                embeddings_persona.append(embedding.cpu().float().numpy().flatten())
                
            except Exception as e:
                print(f"  ⚠️  Error procesando {img_path.name}: {e}")
                continue
        
        if embeddings_persona:
            embeddings_dict[nombre] = embeddings_persona
            print(f"  ✅ {len(embeddings_persona)} embeddings generados")
        else:
            print(f"  ❌ No se generaron embeddings para {nombre}")
    
    # Guardar
    if embeddings_dict:
        with open(output_path, 'wb') as f:
            pickle.dump(embeddings_dict, f)
        
        print("\n" + "="*60)
        print(f"✅ EMBEDDINGS GUARDADOS EN: {output_path}")
        print(f"📊 Total de personas: {len(embeddings_dict)}")
        
        for nombre, embs in embeddings_dict.items():
            print(f"   - {nombre}: {len(embs)} embeddings")
        print("="*60)
    else:
        print("\n❌ No se generaron embeddings")

if __name__ == "__main__":
    generar_embeddings_personas()
