# PERFILADO

Laboratorio docente de inteligencia artificial, identificación facial y protección de datos. **Autor: Antonio Ramos Nieto.**

## Estado de la práctica

- **Introducción:** encargo del equipo técnico y recorrido de la práctica, presentado al final.
- **Paso 1:** una imagen por identidad, nombre, código de perfil y atributos sintéticos. Exploración con búsqueda, filtros, gráficos y preguntas de reflexión.
- **Paso 2:** 5.749 referencias calculadas con detección y alineación facial, y 100 fotos nuevas procesadas en el navegador para buscar top-1/top-5 y ajustar el umbral.
- **Paso 3:** clasificación real de publicaciones RUW con dos objetivos (Rusia/Ucrania), TF-IDF, Naive Bayes o centroides, validación y matriz de confusión.
- **Paso 4:** evaluación conjunta pendiente de desarrollar.

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
textos.html                Paso 3: clasificación de publicaciones
assets/textos.css          Diseño responsive del laboratorio NLP
src/ruw-data.js            Lectura de RUW, limpieza y divisiones
src/text-classifier.js     Vectores TF-IDF y modelos supervisados
src/step3.js               Interfaz y evaluación de dos objetivos
tests/ruw-data.test.js     Integridad del corpus y anonimización parcial
tests/text-classifier.test.js Pruebas de modelos y confusión
scripts/verify-ruw.mjs     Auditoría real de las 6.000 publicaciones
scripts/browser-smoke.mjs  Prueba integrada con Chromium
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

La automatización independiente [Precompute aligned LFW gallery v2 and validate](https://github.com/antoniorani/Perfilado/actions/runs/43620492398) **ha completado correctamente todos sus trabajos** y contrastado las 100 fotografías con la galería completa.

**Resultados reales verificados el 14 de noviembre de 2026:**

- **5.749 fotografías de referencia intentadas**, **5.447 embeddings válidos** y **302 fotografías no detectadas o no procesadas**, excluidas explícitamente.
- **100 fotografías nuevas intentadas**, **99 procesadas**, **1 fallo de detección**.
- **78/99 top-1 correctas (78,8 %)**; top-5 correcto en **85/99 (85,9 %)**.
- Con umbral **0,55**: **78 identificaciones aceptadas correctas**, **20 equivocadas aceptadas**, **1 rechazada** y **1 fotografía no procesada**.
- SHA-256 comprobado de `lfw-gallery-v2.pfg`: `2477d5c7231ee7940f5c501edc8e21e62116ac7d830a202343b2694e21940bab`.

La galería y el informe están en el artefacto **lfw-aligned-v2-all-5749** de la ejecución citada. El informe `lfw-evaluation-100.json` documenta todas las métricas por umbral. El archivo v1 anterior no se puede utilizar con la nueva versión.

### Archivos y uso docente

- `lfw-gallery-v2.pfg`: archivo binario versionado (64 bytes de cabecera + 5.749 × 128 valores `float32`). La longitud íntegra es 2.943.552 bytes. Los registros que no puedan procesarse quedan identificados como `NaN` y se excluyen explícitamente.
- `lfw-gallery-v2-report.json`: identificador de método, SHA-256 de la galería, posiciones que pudieron o no procesarse y trazabilidad del cálculo.
- `lfw-evaluation-100.json`: métricas reales top-1 y por umbral sobre las 100 imágenes de prueba, sin almacenar sus embeddings.

**La galería v1 es incompatible:** no uses `lfw-gallery.f32` de los cálculos anteriores. La página `biometria.html` acepta la galería `lfw-gallery-v3.pfg` una vez actualizada; los modelos anteriores no sirven como sustituto. No basta con renombrar el archivo viejo: es necesario regenerarlo con la detección y alineación nuevas.

Para utilizarlo en clase, descarga el artefacto de GitHub Actions, extrae el archivo `.pfg` y selecciónalo en el campo de carga local del paso 2. El navegador lo conserva únicamente en memoria. La web puede cargarlo automáticamente desde `data/lfw-gallery-v2.pfg` **solo si se revisa y autoriza expresamente su difusión pública**.

**Privacidad:** las plantillas de reconocimiento facial son información biométrica vinculable a identidades reales. Evitamos incorporarlas a la rama pública de GitHub. Los artefactos de Actions tienen una retención limitada, pero también deben tratarse como datos sensibles y gestionarse con las garantías necesarias. No se atribuyen posturas u opiniones políticas a las personas de LFW.

### Límites docentes

- La elección de 100 casos es reproducible; todos aparecen en la galería. **No se mide** la tasa de falsos positivos sobre personas completamente desconocidas.
- La muestra y el modelo no están validados para decisiones operativas. La distancia es una medida geométrica, **no una probabilidad de identidad**.
- El umbral por defecto es didáctico y debe interpretarse con los resultados del modelo v2; cambiarlo no modifica qué identidad ocupa el primer lugar.

## Recuperación de las 302 referencias sin rostro detectado

El experimento v2 generó **5.447 vectores válidos y dejó 302 referencias sin embedding** porque el detector SSD no localizó un rostro con suficiente confianza. El sistema original no tenía una fase de recuperación.

La versión v3 **conserva exactamente los 5.447 embeddings v2** y reintenta únicamente los 302 casos pendientes mediante:

1. SSD MobileNet con umbral de confianza menor, pero manteniendo landmarks y alineación.
2. TinyFaceDetector y los mismos 68 landmarks y red de reconocimiento.
3. SSD aplicado a una imagen ampliada, con alineación de landmarks.
4. Solo si las anteriores fallan: embedding real de la imagen original sin alineación facial confirmada. **Estos casos están identificados como `direct-unverified` y tienen menor fiabilidad**. No deben presentarse como detecciones verificadas.

La galería `lfw-gallery-v3.pfg` usa una cabecera de versión nueva, conserva las posiciones de las 5.749 personas y documenta en `lfw-gallery-v3-report.json` el método utilizado para recuperar cada imagen, así como cualquier fallo residual.

**Recálculo completo verificado: 5.749 vectores finitos / 5.749 identidades.** Los 302 casos se han recuperado con SSD de confianza reducida (**238**), TinyFaceDetector alineado (**59**) y descriptor directo sin alineación confirmada (**5**). Se han conservado bit a bit los **5.447 vectores v2 originales**.

- Artefacto: [lfw-complete-v3-5749, GitHub Actions](https://github.com/antoniorani/Perfilado/actions/runs/43627549177)
- SHA-256 de `lfw-gallery-v3.pfg`: `9f71e3b87b8ad19baad40c9ba1f38b3ce7d23308997b3f2f9a43c55061e3c27c`
- Forma: 5.749 × 128 float32 + 64 bytes de cabecera; **2.943.552 bytes**.

**No se debe interpretar esto como 5.749 identificaciones fiables**: los cinco vectores `direct-unverified` se han generado mediante la red de reconocimiento, pero sin que ningún detector haya confirmado y alineado el rostro. El archivo contiene vectores reales, y el informe documenta expresamente esa diferencia de calidad. Añadir referencias más débiles **no garantiza una mejora en la tasa de aciertos**.

**Atención a la privacidad:** los embeddings generados a partir de caras de personas identificadas constituyen plantillas biométricas que no deben difundirse indiscriminadamente. La web admite la importación local del fichero, sin necesidad de subirlo a GitHub Pages. Los artefactos de GitHub Actions también están sujetos a permisos de acceso y políticas de retención y deben gestionarse adecuadamente.

## Paso 3: clasificación real de publicaciones RUW

**Objetivo:** estudiar qué postura **expresa una publicación** acerca de dos objetivos independientes, Rusia y Ucrania, sin atribuir opiniones privadas a personas identificadas. Las etiquetas originales del corpus [RUW (multi-target/RUW)](https://github.com/multi-target/RUW) son `FAVOR`, `AGAINST` y `NONE`. **`NONE` es ausencia de postura identificable en el mensaje**, no una prueba de neutralidad de la persona.

### Datos de origen y protección de la privacidad

La app carga automáticamente los tres CSV directamente del [repositorio RUW original](https://github.com/multi-target/RUW/tree/ecb2a4579197c5be03092a1b8fa21ead359809ce) a partir del commit fijo `ecb2a4579197c5be03092a1b8fa21ead359809ce`.

| Partición de RUW | Registros originales | Registros utilizados tras descartar duplicados |
| --- | ---: | ---: |
| Entrenamiento | 4.200 | 4.095 |
| Validación (`Dev`) | 600 | 599 |
| Prueba (`Test`) | 1.200 | 1.187 |
| **Total** | **6.000** | **5.881** |

Las coincidencias textuales se excluyen con prioridad **prueba → validación → entrenamiento** para que un ejemplo aparezca como máximo en una partición. Este filtro solo utiliza el texto, no las etiquetas de prueba, y reduce el riesgo de contaminación de la evaluación. Las cifras originales permanecen visibles para que no se oculten los descartes.

**No incluimos en Perfilado una copia de los tuits ni sus autores**. Los textos se obtienen en el navegador desde la fuente y se retiran las menciones de usuario, correos y URL antes de mostrarlos o modelarlos; sigue habiendo contenido de terceros y **no podemos garantizar anonimización completa**. No se vinculan las publicaciones a las personas reales fotografiadas en LFW. El acceso a RUW depende de internet y del servidor de GitHub, y la reutilización del texto debe respetar los derechos y condiciones correspondientes de los autores y de la plataforma.

### Modelos reales y experimento

La aplicación **no usa resultados inventados ni una red neuronal ficticia**:

1. Construye un **vector TF-IDF disperso** de palabras (o palabras y pares) con el conjunto de entrenamiento. Es una representación léxica y **no** un embedding semántico neuronal tipo BERT.
2. Entrena en el navegador dos clasificadores alternativos: **Naive Bayes multinomial** sobre TF-IDF o **clasificador de centroides por similitud coseno**.
3. Ajusta objetivo (Rusia/Ucrania), representación, tamaño de vocabulario y suavizado (Naive Bayes).
4. Muestra exactitud, **F1 macro**, matriz de confusión, métricas por clase, términos del modelo y errores del conjunto de **validación**.
5. Tras elegir configuración, desbloquea una evaluación separada del **conjunto de prueba reservado**, con su matriz de confusión. Se advierte que utilizar resultados de prueba para volver a ajustar parámetros introduce sesgo.
6. Permite analizar una frase escrita por el alumno, exclusivamente en memoria local.

En la auditoría con el corpus original, vocabulario de 2.500 términos y unigramas, se obtuvieron las siguientes métricas de validación **reales**, no extrapolables a opiniones de personas:

| Objetivo | Clasificador | Exactitud | F1 macro |
| --- | --- | ---: | ---: |
| Rusia | Naive Bayes | 50,8 % | 37,3 % |
| Rusia | Centroides por coseno | 50,3 % | 48,8 % |
| Ucrania | Naive Bayes | 53,6 % | 47,2 % |
| Ucrania | Centroides por coseno | 52,3 % | 51,5 % |

Estos modelos son deliberadamente manejables para un taller en el navegador. No replican los algoritmos avanzados ni las cifras del artículo RUW; ofrecen una **línea de base reproducible**. La exactitud puede parecer superior aun cuando algunas clases tengan una sensibilidad baja: por eso incluimos F1 macro y matriz de confusión.

Las pruebas automatizadas ejecutan `npm test`; para auditar la fuente RUW con conexión a internet, ejecuta `node scripts/verify-ruw.mjs`. La integración real del navegador se valida mediante Chromium en `scripts/browser-smoke.mjs`. Los datos no se envían a una API propia de Perfilado.

## GitHub Pages

En **Settings → Pages → Build and deployment**, selecciona **Deploy from a branch → main → / (root)**. Una vez habilitado el despliegue:

https://antoniorani.github.io/Perfilado/

Las rutas relativas funcionan bajo el subdirectorio `/Perfilado/`.

**Uso exclusivamente docente.** La app no opera un sistema de vigilancia y no ejecuta aún modelos de procesamiento del lenguaje. La etapa conjunta sobre opiniones políticas deberá construirse con identidades totalmente ficticias.
