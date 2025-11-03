#!/usr/bin/env python3
"""
Script para registrar rostros y extraer embeddings usando DeepFace.
Procesa todas las imágenes en la carpeta photos/training y guarda los embeddings.
"""

import os
import pickle
from pathlib import Path
from deepface import DeepFace
import numpy as np

# Configuración
PHOTOS_DIR = "photos/training"
EMBEDDINGS_DIR = "embeddings"
EMBEDDINGS_FILE = os.path.join(EMBEDDINGS_DIR, "face_embeddings.pkl")

# Modelo a usar (opciones: VGG-Face, Facenet, Facenet512, OpenFace, DeepFace, DeepID, ArcFace, Dlib, SFace)
MODEL_NAME = "Facenet512"  # Facenet512 es muy preciso y rápido

def extract_embeddings():
    """
    Extrae embeddings de todas las fotos en la carpeta de entrenamiento.
    """
    print(f"🔍 Buscando imágenes en: {PHOTOS_DIR}")

    # Verificar que existe la carpeta
    if not os.path.exists(PHOTOS_DIR):
        print(f"❌ Error: La carpeta {PHOTOS_DIR} no existe")
        return

    # Buscar todas las imágenes
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp'}
    image_files = []

    for file in os.listdir(PHOTOS_DIR):
        if Path(file).suffix.lower() in image_extensions:
            image_files.append(os.path.join(PHOTOS_DIR, file))

    if not image_files:
        print(f"⚠️  No se encontraron imágenes en {PHOTOS_DIR}")
        print(f"   Por favor, agrega fotos con extensiones: {', '.join(image_extensions)}")
        return

    print(f"📸 Encontradas {len(image_files)} imágenes")

    # Diccionario para almacenar embeddings
    # Formato: {"person_id": {"embedding": array, "photo_path": str}}
    embeddings_db = {}

    # Procesar cada imagen
    for idx, img_path in enumerate(image_files, 1):
        try:
            print(f"\n[{idx}/{len(image_files)}] Procesando: {os.path.basename(img_path)}")

            # Extraer embedding usando DeepFace
            # DeepFace.represent retorna una lista de diccionarios con 'embedding' y 'facial_area'
            result = DeepFace.represent(
                img_path=img_path,
                model_name=MODEL_NAME,
                enforce_detection=True,  # Falla si no detecta un rostro
                detector_backend="opencv",  # Opciones: opencv, ssd, dlib, mtcnn, retinaface
                align=True
            )

            # Tomar el primer rostro detectado
            embedding = result[0]["embedding"]

            # Usar el nombre del archivo (sin extensión) como ID de persona
            person_id = Path(img_path).stem

            embeddings_db[person_id] = {
                "embedding": np.array(embedding),
                "photo_path": img_path,
                "model": MODEL_NAME
            }

            print(f"   ✅ Embedding extraído (dimensión: {len(embedding)})")

        except Exception as e:
            print(f"   ❌ Error procesando {os.path.basename(img_path)}: {str(e)}")
            continue

    # Guardar embeddings
    if embeddings_db:
        os.makedirs(EMBEDDINGS_DIR, exist_ok=True)

        with open(EMBEDDINGS_FILE, 'wb') as f:
            pickle.dump(embeddings_db, f)

        print(f"\n✅ Embeddings guardados exitosamente!")
        print(f"   Archivo: {EMBEDDINGS_FILE}")
        print(f"   Total de personas registradas: {len(embeddings_db)}")
        print(f"   IDs registrados: {', '.join(embeddings_db.keys())}")
    else:
        print("\n⚠️  No se pudo extraer ningún embedding")

def list_registered_faces():
    """
    Lista los rostros registrados en la base de datos.
    """
    if not os.path.exists(EMBEDDINGS_FILE):
        print("⚠️  No hay rostros registrados aún")
        return

    with open(EMBEDDINGS_FILE, 'rb') as f:
        embeddings_db = pickle.load(f)

    print(f"\n📋 Rostros registrados: {len(embeddings_db)}")
    for person_id, data in embeddings_db.items():
        print(f"   - {person_id} (modelo: {data.get('model', 'unknown')})")

if __name__ == "__main__":
    print("=" * 60)
    print("   REGISTRO DE ROSTROS - DeepFace")
    print("=" * 60)

    # Extraer embeddings
    extract_embeddings()

    # Mostrar rostros registrados
    list_registered_faces()

    print("\n" + "=" * 60)
    print("✨ Proceso completado")
    print("=" * 60)
