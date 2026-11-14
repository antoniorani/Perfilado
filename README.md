# PERFILADO

Laboratorio docente de inteligencia artificial, identificación facial y protección de datos. **Autor: Antonio Ramos Nieto.**

## Estado de la práctica

- **Introducción:** encargo del equipo técnico y recorrido de la práctica, presentado al final.
- **Paso 1:** una imagen por identidad, nombre, código de perfil y atributos sintéticos. Exploración con búsqueda, filtros, gráficos y preguntas de reflexión.
- **Paso 2:** 5.749 referencias calculadas con detección y alineación facial, y 100 fotos nuevas procesadas en el navegador para buscar top-1/top-5 y ajustar el umbral.
- **Pasos 3–4:** análisis NLP y evaluación conjunta. Pendientes de desarrollar.

## Datos

La web utiliza el [dataset Labeled Faces in the Wild (LFW)](https://vis-www.cs.umass.edu/lfw/), publicado por investigadores de la Universidad de Massachusetts Amherst (2007), con 5.749 identidades y 13.233 imágenes. El fichero [`data/lfw-identities.json`](data/lfw-identities.json) contiene **únicamente los nombres originales y el número de imágenes** por identidad, contrastados con [`lfw-names.txt`](https://github.com/happynear/FaceDatasets/blob/master/CASIA/lfw-names.txt).

La fotografía inicial (`Nombre_Apellido_0001.jpg`) de cada perfil **se visualiza automáticamente** desde la copia pública [marcelohaps/lfw de Hugging Face](https://huggingface.co/datasets/marcelohaps/lfw). Solo se solicitan las diez fotografías que aparecen en la página actual; no se descargan previamente las 5.749 ni se almacenan fotografías en este repositorio. Si la conexión o el proveedor externo fallan, la interfaz muestra una imagen no disponible. La visualización necesita acceso a internet y realiza solicitudes al proveedor externo.

**Nombre e imagen proceden de LFW.** El código del perfil, la edad, la zona y el idioma son **ficticios**, generados con una función determinista. La distribución de zonas es deliberadamente desigual, pero **no es una distribución real ni representativa de LFW**. Nunca se atribuyen opiniones inventadas a las personas reales fotografiadas.

Este proyecto no habilita la reutilización libre de las fotografías para otras finalidades; cualquier uso externo requiere revisar derechos, licencias y protección de datos. La documentación de Hugging Face señala que se deben comprobar las condiciones de origen antes de redistribuir las imágenes.

## Código y pruebas

Web estática sin dependencias: HTML, CSS y módulos de JavaScript. No necesita backend, instalación ni compilación.

```text
index.html                 Introducción
datos.html                 Paso 1
assets/styles.css          Diseño adaptable
src/core.js                Transformaciones puras
src/step1.js               UI de exploración
biometria.html             Paso 2: identificación biométrica
assets/biometrics.css      Estilos específicos del laboratorio
src/answers.js             Respuestas compartidas entre pasos
src/face-engine.js         Modelos preentrenados reales
src/face-pipeline.js       Pipeline compartido de detección, alineación y embeddings
src/gallery-format.js      Cabecera versionada y validación del catálogo
src/step2.js               Experimento real de 100 fotografías
src/full-gallery.js        Selección reproducible, lectura f32 y búsqueda top-5
scripts/build-gallery.mjs   Generación reproducible de embeddings LFW
scripts/combine-gallery.mjs Ensamblado versionado de los lotes
scripts/node-face-model.mjs Preparación común de modelos Node
scripts/validate-alignment.mjs Validación de 24 identidades reales
scripts/evaluate-gallery.mjs Evaluación de 100 fotos con 5.749 referencias
.github/workflows/precompute-gallery.yml  Proceso de precálculo
data/lfw-identities.json   Índice real LFW
tests/core.test.js         Pruebas del catálogo
tests/full-gallery.test.js   Pruebas de búsqueda y umbrales
```

Para trabajar localmente:

```bash
python3 -m http.server 8000
# abrir http://localhost:8000
```

Para probar la lógica (Node.js >=20):

```bash
npm test
```

Las respuestas se guardan localmente en el navegador, si el almacenamiento está disponible, y pueden exportarse a TXT. **La interfaz ya no ofrece descarga del conjunto de perfiles**.

## Paso 2: identificación con detección y alineación real

### Problema detectado y solución

La primera versión (v1) calculaba descriptores pasando imágenes completas a `computeFaceDescriptor(image)`, sin detección y alineación. Los resultados observados (8 % top-1 en 100 fotos) no deben utilizarse como una medición válida del reconocimiento facial.

La versión **v2** utiliza la misma función `computeAlignedDescriptor()` en Node y en el navegador:

1. Detector **SSD MobileNet V1** sobre la fotografía original de LFW.
2. Detección de **68 puntos faciales** y alineación automática.
3. Embedding de 128 componentes con **faceRecognitionNet**, usando pesos de `face-api.js@0.22.2`.
4. Distancia euclidiana de cada una de las 100 fotografías nuevas a la galería de referencias.
5. Evaluación separada de **top-1** (independiente del umbral), aciertos aceptados, errores aceptados, rechazos y fotos que no pudieron procesarse.

Se usa el mismo módulo `src/face-pipeline.js` para los dos entornos. Para evitar comparar vectores de métodos diferentes, el archivo de referencias tiene ahora una cabecera con versión del modelo, tamaño y huella del catálogo, definida en `src/gallery-format.js`. Los archivos anteriores `.f32` se rechazan explícitamente.

Además, la web vuelve a calcular un embedding de control a partir de una imagen ya presente en la galería y verifica su concordancia **antes de evaluar las 100 nuevas**.

### Validación previa

Se ha ejecutado una prueba real sobre **24 identidades con dos fotografías de LFW cada una**: **24/24 coincidencias top-1** y ningún fallo de procesamiento dentro de esa muestra de 24 referencias. Este resultado comprueba que el procedimiento nuevo funciona mejor en esa muestra, pero **no equivale a una estimación sobre las 5.749 referencias**.

La automatización independiente `Precompute aligned LFW gallery v2 and validate` calcula las primeras fotografías de 5.749 personas por lotes y, después de ensamblar y verificar los índices, evalúa otras **100 fotografías contra la galería completa**. El informe resultante `lfw-evaluation-100.json` permitirá distinguir los resultados del estudio de 24 personas de los del ejercicio completo.

### Archivos y uso docente

- `lfw-gallery-v2.pfg`: archivo binario versionado (64 bytes de cabecera + 5.749 × 128 valores `float32`). La longitud íntegra es 2.943.552 bytes. Los registros que no puedan procesarse quedan identificados como `NaN` y se excluyen explícitamente.
- `lfw-gallery-v2-report.json`: identificador de método, SHA-256 de la galería, posiciones que pudieron o no procesarse y trazabilidad del cálculo.
- `lfw-evaluation-100.json`: métricas reales top-1 y por umbral sobre las 100 imágenes de prueba, sin almacenar sus embeddings.

**La galería v1 es incompatible:** no uses `lfw-gallery.f32` de los cálculos anteriores. La página `biometria.html` acepta exclusivamente `lfw-gallery-v2.pfg`. No basta con renombrar el archivo viejo: es necesario regenerarlo con la detección y alineación nuevas.

Para utilizarlo en clase, descarga el artefacto de GitHub Actions, extrae el archivo `.pfg` y selecciónalo en el campo de carga local del paso 2. El navegador lo conserva únicamente en memoria. La web puede cargarlo automáticamente desde `data/lfw-gallery-v2.pfg` **solo si se revisa y autoriza expresamente su difusión pública**.

**Privacidad:** las plantillas de reconocimiento facial son información biométrica vinculable a identidades reales. Evitamos incorporarlas a la rama pública de GitHub. Los artefactos de Actions tienen una retención limitada, pero también deben tratarse como datos sensibles y gestionarse con las garantías necesarias. No se atribuyen posturas u opiniones políticas a las personas de LFW.

### Límites docentes

- La elección de 100 casos es reproducible; todos aparecen en la galería. **No se mide** la tasa de falsos positivos sobre personas completamente desconocidas.
- La muestra y el modelo no están validados para decisiones operativas. La distancia es una medida geométrica, **no una probabilidad de identidad**.
- El umbral por defecto es didáctico y debe interpretarse con los resultados del modelo v2; cambiarlo no modifica qué identidad ocupa el primer lugar.

## GitHub Pages

En **Settings → Pages → Build and deployment**, selecciona **Deploy from a branch → main → / (root)**. Una vez habilitado el despliegue:

https://antoniorani.github.io/Perfilado/

Las rutas relativas funcionan bajo el subdirectorio `/Perfilado/`.

**Uso exclusivamente docente.** La app no opera un sistema de vigilancia y no ejecuta aún modelos de procesamiento del lenguaje. La etapa conjunta sobre opiniones políticas deberá construirse con identidades totalmente ficticias.
