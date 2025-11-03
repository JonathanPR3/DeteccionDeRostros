#!/usr/bin/env python3
"""
Script para reconocimiento facial en tiempo real usando webcam.
Compara rostros detectados con los embeddings almacenados.
"""

import os
import pickle
import cv2
import numpy as np
from deepface import DeepFace
from scipy.spatial.distance import cosine

# Configuración
EMBEDDINGS_FILE = "embeddings/face_embeddings.pkl"
MODEL_NAME = "Facenet512"
THRESHOLD = 0.4  # Umbral de similitud (menor = más estricto, rango 0-1)

# Colores para el texto (BGR)
COLOR_RECOGNIZED = (0, 255, 0)  # Verde
COLOR_UNKNOWN = (0, 0, 255)  # Rojo
COLOR_PROCESSING = (255, 255, 0)  # Cyan

class FaceRecognizer:
    def __init__(self, embeddings_file, model_name, threshold):
        self.embeddings_file = embeddings_file
        self.model_name = model_name
        self.threshold = threshold
        self.embeddings_db = {}
        self.load_embeddings()

    def detect_available_cameras(self, max_cameras=5):
        """Detecta las cámaras disponibles."""
        available_cameras = []

        print("\n🔍 Buscando cámaras disponibles...")
        for i in range(max_cameras):
            cap = cv2.VideoCapture(i)
            if cap.isOpened():
                ret, _ = cap.read()
                if ret:
                    available_cameras.append(i)
                    print(f"   ✅ Cámara {i} detectada")
                cap.release()

        return available_cameras

    def select_camera_source(self):
        """Permite al usuario seleccionar la fuente de la cámara."""
        print("\n" + "=" * 60)
        print("   SELECCIÓN DE CÁMARA")
        print("=" * 60)

        # Detectar cámaras USB disponibles
        available_cameras = self.detect_available_cameras()

        print("\n📹 Opciones disponibles:")
        print("   1. Cámara USB/integrada (índice)")
        print("   2. Webcam WiFi (URL)")

        if available_cameras:
            print(f"\n   Cámaras USB detectadas: {available_cameras}")

        choice = input("\nSelecciona opción (1 o 2): ").strip()

        if choice == "1":
            if available_cameras:
                print(f"\nCámaras disponibles: {available_cameras}")
                camera_idx = input(f"Ingresa el índice de cámara [{available_cameras[0]}]: ").strip()
                camera_idx = int(camera_idx) if camera_idx else available_cameras[0]
            else:
                camera_idx = input("Ingresa el índice de cámara [0]: ").strip()
                camera_idx = int(camera_idx) if camera_idx else 0
            return camera_idx

        elif choice == "2":
            print("\n💡 Ejemplos de URLs:")
            print("   - HTTP: http://192.168.1.100:8080/video")
            print("   - RTSP: rtsp://192.168.1.100:8554/stream")
            url = input("\nIngresa la URL de la webcam WiFi: ").strip()
            return url

        else:
            print("⚠️  Opción inválida, usando cámara 0 por defecto")
            return 0

    def load_embeddings(self):
        """Carga los embeddings de rostros registrados."""
        if not os.path.exists(self.embeddings_file):
            print(f"❌ Error: No se encontró {self.embeddings_file}")
            print("   Ejecuta primero 'enroll_faces.py' para registrar rostros")
            exit(1)

        with open(self.embeddings_file, 'rb') as f:
            self.embeddings_db = pickle.load(f)

        print(f"✅ Cargados {len(self.embeddings_db)} rostros registrados")
        print(f"   IDs: {', '.join(self.embeddings_db.keys())}")

    def find_match(self, embedding):
        """
        Encuentra el rostro más similar en la base de datos.
        Retorna: (person_id, similarity_score) o (None, None) si no hay match
        """
        best_match = None
        best_distance = float('inf')

        for person_id, data in self.embeddings_db.items():
            stored_embedding = data["embedding"]

            # Calcular distancia coseno (0 = idéntico, 2 = opuesto)
            distance = cosine(embedding, stored_embedding)

            if distance < best_distance:
                best_distance = distance
                best_match = person_id

        # Si la distancia es menor al umbral, es un match
        if best_distance < self.threshold:
            similarity = 1 - best_distance  # Convertir a similitud (0-1)
            return best_match, similarity
        else:
            return None, None

    def run_webcam_recognition(self):
        """Ejecuta reconocimiento facial en tiempo real con webcam."""
        print(f"\n⚙️  Modelo: {self.model_name}")
        print(f"⚙️  Umbral de reconocimiento: {self.threshold}")

        # Seleccionar fuente de cámara
        camera_source = self.select_camera_source()

        print("\n🎥 Iniciando cámara...")
        print("\n📌 Presiona 'q' para salir\n")

        # Inicializar cámara con la fuente seleccionada
        cap = cv2.VideoCapture(camera_source)

        if not cap.isOpened():
            print(f"❌ Error: No se pudo abrir la cámara")
            print(f"   Fuente intentada: {camera_source}")
            print("\n💡 Sugerencias:")
            print("   - Verifica que la URL de la webcam WiFi sea correcta")
            print("   - Asegúrate que la webcam WiFi esté encendida y conectada")
            print("   - Prueba con un índice diferente (0, 1, 2...)")
            return

        # Variables para control de procesamiento
        frame_count = 0
        process_every_n_frames = 10  # Procesar cada 10 frames para mejor rendimiento

        # Estado persistente del último resultado detectado
        last_detection = None  # Guardará: {"facial_area": dict, "label": str, "color": tuple}

        while True:
            ret, frame = cap.read()
            if not ret:
                print("❌ Error al capturar frame")
                break

            frame_count += 1
            display_frame = frame.copy()

            # Procesar solo cada N frames
            if frame_count % process_every_n_frames == 0:
                try:
                    # Detectar y extraer embedding del rostro
                    result = DeepFace.represent(
                        img_path=frame,
                        model_name=self.model_name,
                        enforce_detection=False,  # No fallar si no detecta rostro
                        detector_backend="opencv",
                        align=True
                    )

                    # Si se detectó al menos un rostro
                    if result and len(result) > 0:
                        # Procesar el primer rostro detectado
                        embedding = np.array(result[0]["embedding"])
                        facial_area = result[0]["facial_area"]

                        # Buscar coincidencia
                        person_id, similarity = self.find_match(embedding)

                        # Preparar información de detección
                        x = facial_area["x"]
                        y = facial_area["y"]
                        w = facial_area["w"]
                        h = facial_area["h"]

                        if person_id:
                            # Rostro reconocido
                            color = COLOR_RECOGNIZED
                            label = f"{person_id} ({similarity*100:.1f}%)"
                        else:
                            # Rostro desconocido
                            color = COLOR_UNKNOWN
                            label = "DESCONOCIDO"

                        # Guardar detección para mantenerla visible
                        last_detection = {
                            "facial_area": facial_area,
                            "label": label,
                            "color": color
                        }
                    else:
                        # No se detectó rostro, limpiar detección previa
                        last_detection = None

                except Exception as e:
                    # Si hay error (ej: no se detecta rostro), limpiar detección
                    last_detection = None

            # Dibujar la última detección válida en TODOS los frames
            if last_detection:
                facial_area = last_detection["facial_area"]
                label = last_detection["label"]
                color = last_detection["color"]

                x = facial_area["x"]
                y = facial_area["y"]
                w = facial_area["w"]
                h = facial_area["h"]

                # Dibujar rectángulo
                cv2.rectangle(display_frame, (x, y), (x + w, y + h), color, 2)

                # Fondo para el texto
                label_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
                cv2.rectangle(display_frame,
                            (x, y - 30),
                            (x + label_size[0], y),
                            color,
                            cv2.FILLED)

                # Texto
                cv2.putText(display_frame,
                          label,
                          (x, y - 10),
                          cv2.FONT_HERSHEY_SIMPLEX,
                          0.6,
                          (255, 255, 255),
                          2)

            # Mostrar información en pantalla
            info_text = f"Rostros registrados: {len(self.embeddings_db)} | Presiona 'q' para salir"
            cv2.putText(display_frame,
                       info_text,
                       (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX,
                       0.6,
                       (255, 255, 255),
                       2)

            # Mostrar frame
            cv2.imshow('Reconocimiento Facial - DeepFace', display_frame)

            # Salir con 'q'
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        # Limpiar
        cap.release()
        cv2.destroyAllWindows()
        print("\n👋 Webcam cerrada")

if __name__ == "__main__":
    print("=" * 60)
    print("   RECONOCIMIENTO FACIAL EN TIEMPO REAL")
    print("=" * 60)

    recognizer = FaceRecognizer(
        embeddings_file=EMBEDDINGS_FILE,
        model_name=MODEL_NAME,
        threshold=THRESHOLD
    )

    recognizer.run_webcam_recognition()

    print("=" * 60)
    print("✨ Proceso completado")
    print("=" * 60)
