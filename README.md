# Sistema de Reconocimiento Facial con DeepFace

Sistema de reconocimiento facial en tiempo real que utiliza embeddings faciales para identificar personas a través de webcam. Diseñado como prueba de concepto para integración con aplicaciones móviles.

## Características

- **Reconocimiento en tiempo real** mediante webcam (USB o WiFi)
- **Almacenamiento eficiente** usando embeddings en lugar de imágenes
- **Alta precisión** con modelo Facenet512
- **Escalable** para integración con APIs
- **Privacidad mejorada** - los embeddings no pueden revertirse a imágenes originales
- **Interfaz visual** con indicadores de confianza en porcentaje

## Cómo Funciona

### Arquitectura del Sistema

```
1. Registro (Enrollment):
   Foto del usuario → DeepFace → Embedding (vector 512D) → Almacenamiento (PKL)

2. Reconocimiento:
   Webcam → DeepFace → Embedding → Comparación (cosine distance) → Identificación
```

### Ventajas del Enfoque con Embeddings

- **Privacidad**: Los embeddings son vectores matemáticos que no pueden revertirse a la imagen original
- **Eficiencia**: ~512 bytes por persona vs varios KB/MB de imágenes
- **Velocidad**: Comparación de vectores es extremadamente rápida
- **Escalabilidad**: Compatible con bases de datos vectoriales (FAISS, Pinecone, Milvus)

## Requisitos

- Python 3.8+
- Webcam (USB integrada o WiFi)
- Sistema operativo: Windows, macOS o Linux

## Instalación

### 1. Clonar el repositorio

```bash
git clone <repository-url>
cd face_recognition_project
```

### 2. Crear entorno virtual (recomendado)

```bash
python -m venv venv

# Windows
venv\Scripts\activate

# macOS/Linux
source venv/bin/activate
```

### 3. Instalar dependencias

```bash
pip install -r requirements.txt
```

### 4. Crear estructura de carpetas

```bash
mkdir photos\training
mkdir embeddings
```

## Uso

### Paso 1: Registrar Rostros (Enrollment)

1. **Agregar fotos de entrenamiento**

   Coloca imágenes en `photos/training/`. El nombre del archivo será el ID de la persona:

   ```
   photos/training/
   ├── jonathan.jpg      → ID: "jonathan"
   ├── maria.jpg         → ID: "maria"
   └── carlos.png        → ID: "carlos"
   ```

   **Recomendaciones para las fotos:**
   - Buena iluminación
   - Rostro frontal y visible
   - Sin lentes oscuros ni mascarillas
   - Resolución mínima: 640x480px
   - Formatos soportados: JPG, PNG, BMP

2. **Extraer embeddings**

   ```bash
   python enroll_faces.py
   ```

   Esto generará `embeddings/face_embeddings.pkl` con los vectores de cada persona.

### Paso 2: Reconocimiento en Tiempo Real

```bash
python recognize_webcam.py
```

**Opciones de cámara:**
- Opción 1: Cámara USB/integrada (índice: 0, 1, 2...)
- Opción 2: Webcam WiFi (URL: http://192.168.x.x:8080/video)

**Controles:**
- `q` - Salir del modo reconocimiento

**Indicadores visuales:**
- Cuadro **verde** + nombre + confianza → Persona reconocida
- Cuadro **rojo** + "DESCONOCIDO" → Rostro no identificado

## Estructura del Proyecto

```
face_recognition_project/
│
├── photos/
│   └── training/              # Fotos de personas a registrar
│       ├── persona1.jpg
│       └── persona2.jpg
│
├── embeddings/                # Base de datos de embeddings
│   └── face_embeddings.pkl
│
├── enroll_faces.py           # Script de registro de rostros
├── recognize_webcam.py       # Script de reconocimiento en tiempo real
├── requirements.txt          # Dependencias del proyecto
└── README.md                 # Este archivo
```

## Configuración

### Ajustar Parámetros

Puedes modificar estos parámetros en `recognize_webcam.py`:

```python
# Línea 16-17
MODEL_NAME = "Facenet512"     # Modelo de embedding
THRESHOLD = 0.4               # Umbral de similitud

# process_every_n_frames = 10  # Línea 150 - Frames a procesar
```

### Threshold de Reconocimiento

El `THRESHOLD` controla qué tan estricto es el reconocimiento:

- `0.3` - **Más estricto** (menos falsos positivos, puede rechazar personas válidas)
- `0.4` - **Balanceado** (recomendado)
- `0.5` - **Más permisivo** (menos falsos negativos, puede aceptar personas incorrectas)

### Modelos Disponibles

Puedes cambiar el modelo en ambos scripts (deben usar el mismo):

| Modelo | Precisión | Velocidad | Dimensión |
|--------|-----------|-----------|-----------|
| Facenet512 | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐ | 512D |
| VGG-Face | ⭐⭐⭐⭐ | ⭐⭐⭐ | 2622D |
| ArcFace | ⭐⭐⭐⭐⭐ | ⭐⭐⭐ | 512D |
| Facenet | ⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | 128D |

## Uso con Webcam WiFi

Si usas una app de webcam WiFi (ej: IP Webcam, DroidCam):

1. Instala la app en tu smartphone
2. Conecta el teléfono a la misma red WiFi que tu PC
3. Anota la URL que muestra la app (ej: `http://192.168.1.100:8080/video`)
4. Ejecuta `python recognize_webcam.py` y selecciona opción 2
5. Ingresa la URL

## Próximos Pasos / Roadmap

### Mejoras Planificadas

- [ ] API REST para integración con apps móviles
- [ ] Soporte para múltiples embeddings por usuario (diferentes ángulos/iluminación)
- [ ] Base de datos relacional (PostgreSQL) + vectorial (FAISS)
- [ ] Detección de vivacidad (liveness detection) contra spoofing
- [ ] Reconocimiento de múltiples rostros simultáneos
- [ ] Dashboard web para gestión de usuarios
- [ ] Logs y métricas de confianza

### Integración con App Móvil (Concepto)

```
[App Móvil] → POST /api/enroll
              ├── Imagen del usuario
              └── user_id

[Backend] → Extrae embedding → Guarda en BD

[App Móvil] → POST /api/verify
              └── Imagen para verificar

[Backend] → Extrae embedding → Compara → Retorna match
```

## Troubleshooting

### Error: No se detecta la cámara

```bash
# Listar cámaras disponibles
# El script automáticamente detecta cámaras, pero puedes probar índices manualmente
# Índices comunes: 0 (integrada), 1 (USB externa)
```

### Error: "No face detected"

- Asegúrate de tener buena iluminación
- Verifica que tu rostro esté completamente visible
- Prueba con el detector `mtcnn` (más lento pero más preciso):
  ```python
  detector_backend="mtcnn"  # En lugar de "opencv"
  ```

### Error: Falsos positivos/negativos

- **Ajusta el THRESHOLD** (ver sección Configuración)
- Agrega más fotos de entrenamiento con diferentes condiciones
- Considera múltiples embeddings por usuario

### Performance lento

- Reduce `process_every_n_frames` a 5-15 (línea 150)
- Usa un modelo más rápido: `Facenet` en lugar de `Facenet512`
- Reduce resolución de la cámara

## Mejores Prácticas

### Para Registro de Rostros

1. **Múltiples fotos por persona**: Captura 2-3 fotos con diferentes:
   - Ángulos (frontal, leve giro izquierda/derecha)
   - Condiciones de luz (natural, artificial)
   - Expresiones (neutral, sonrisa)

2. **Calidad de imagen**:
   - Mínimo 640x480px
   - Rostro ocupa al menos 30% del frame
   - Sin blur o desenfoque

3. **Condiciones consistentes**:
   - Fondo neutro preferible
   - Evitar sombras duras
   - Sin obstrucciones (pelo, manos, objetos)

### Para Producción

1. **Versionado de embeddings**: Guarda el modelo usado
2. **Threshold por caso de uso**:
   - Seguridad alta (acceso): 0.3
   - Personalización (UX): 0.5
3. **Re-entrenamiento**: Actualizar embeddings cada 6-12 meses
4. **Logs**: Registrar confianza y matches para análisis

## Consideraciones de Seguridad y Privacidad

- Los embeddings no pueden revertirse a imágenes originales
- No almacenar imágenes originales en producción
- Implementar liveness detection contra ataques de replay
- Cumplir con regulaciones de datos biométricos (GDPR, CCPA, etc.)
- Encriptar embeddings en almacenamiento
- Usar HTTPS para transferencia de datos

## Tecnologías Utilizadas

- **DeepFace**: Framework de reconocimiento facial
- **Facenet512**: Modelo de embeddings faciales
- **OpenCV**: Procesamiento de video y detección facial
- **NumPy**: Operaciones con arrays
- **SciPy**: Cálculo de distancia coseno

## Contribuciones

Este es un proyecto de prueba de concepto. Sugerencias y mejoras son bienvenidas.

## Licencia

[Especificar licencia]

## Contacto

[Tu información de contacto o del equipo]

---

**Nota**: Este proyecto está diseñado como prueba de concepto. Para uso en producción, considera agregar autenticación, manejo robusto de errores, y pruebas exhaustivas.
