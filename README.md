# PERFILADO

Laboratorio docente de inteligencia artificial, identificación facial y protección de datos. **Autor: Antonio Ramos Nieto.**

## Estado de la práctica

- **Introducción:** encargo del equipo técnico y recorrido de la práctica, presentado al final.
- **Paso 1:** una imagen por identidad, nombre, código de perfil y atributos sintéticos. Exploración con búsqueda, filtros, gráficos y preguntas de reflexión.
- **Paso 2:** 5.749 referencias calculadas con detección y alineación facial, y 100 fotos nuevas procesadas en el navegador para buscar top-1/top-5 y ajustar el umbral.
- **Paso 3:** clasificación real de RUW con dos objetivos (Rusia/Ucrania), tres clasificadores entrenados **en el navegador**, incluido MiniLM como extractor de embeddings semánticos y una cabeza logística supervisada, con validación, prueba y matriz de confusión.
- **Paso 4:** evaluación conjunta de los errores de los pasos 2 y 3, inferencias sobre opiniones políticas, escenarios jurídicos civil/policial/militar y valoración final. Todos los casos son ficticios.

## Datos

La web utiliza el [dataset Labeled Faces in the Wild (LFW)](https://vis-www.cs.umass.edu/lfw/), publicado por investigadores de la Universidad de Massachusetts Amherst (2007), con 5.749 identidades y 13.233 imágenes. El fichero [`data/lfw-identities.json`](data/lfw-identities.json) contiene **únicamente los nombres originales y el número de imágenes** por identidad, contrastados con [`lfw-names.txt`](https://github.com/happynear/FaceDatasets/blob/master/CASIA/lfw-names.txt).

La fotografía inicial (`Nombre_Apellido_0001.jpg`) de cada perfil **se visualiza automáticamente** desde la copia pública [marcelohaps/lfw de Hugging Face](https://huggingface.co/datasets/marcelohaps/lfw). Solo se solicitan las diez fotografías que aparecen en la página actual; no se descargan previamente las 5.749 ni se almacenan fotografías en este repositorio. Si la conexión o el proveedor externo fallan, la interfaz muestra una imagen no disponible. La visualización necesita acceso a internet y realiza solicitudes al proveedor externo.

**Nombre e imagen proceden de LFW.** El código del perfil, la edad, la zona y el idioma son **ficticios**, generados con una función determinista. La distribución de zonas es deliberadamente desigual, pero **no es una distribución real ni representativa de LFW**. Nunca se atribuyen opiniones inventadas a las personas reales fotografiadas.

Este proyecto no habilita la reutilización libre de las fotografías para otras finalidades; cualquier uso externo requiere revisar derechos, licencias y protección de datos. La documentación de Hugging Face señala que se deben comprobar las condiciones de origen antes de redistribuir las imágenes.

## Código y pruebas

Web estática sin backend, instalación ni compilación: HTML, CSS y módulos de JavaScript. La opción semántica descarga Transformers.js y el encoder MiniLM desde CDN/Hugging Face en su primer uso y ejecuta ONNX dentro del navegador.

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
evaluacion.html            Paso 4: evaluación conjunta e informe final (preguntas 20–23)
assets/evaluacion.css      Estilos de la fase final
src/combined-risk.js       Producto hipotético de aciertos y límites sin independencia
src/step4.js               Simulación local y respuestas reutilizando src/answers.js
tests/combined-risk.test.js Pruebas de cálculos y extremos
assets/textos.css          Diseño responsive del laboratorio NLP
src/ruw-data.js            Lectura de RUW, limpieza y divisiones
src/text-classifier.js     Vectores TF-IDF, métricas compartidas y modelos léxicos
src/embedding-classifier.js  Regresión logística entrenada en el navegador
src/semantic-worker.js    Embeddings MiniLM/ONNX en Web Worker
src/ruw-ridge.js           Modelo Ridge histórico (ya no usado en la interfaz)
data/ruw-joint-ridge-v1.json Modelo aprendido, sin publicaciones de RUW
scripts/export-ruw-joint-model.py Entrenamiento reproducible con sklearn
scripts/verify-ruw-ridge.mjs Paridad de predicciones JS/sklearn
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

**No almacenamos ni versionamos una copia de los tuits ni sus autores en este repositorio**. Los textos se descargan desde la fuente y se muestran temporalmente en el navegador, retirando menciones de usuario, correos y URL antes de mostrarlos o modelarlos. Son publicaciones de terceros, algunas con lenguaje ofensivo o información no verificada, y **no podemos garantizar anonimización completa**. No se vinculan las publicaciones a las personas reales fotografiadas en LFW. El acceso a RUW depende de internet y del servidor de GitHub, y la reutilización del texto debe respetar los derechos y condiciones correspondientes de los autores y de la plataforma.

### Modelos reales y experimento

La aplicación **no usa puntuaciones inventadas**. En el paso 3 todos los clasificadores seleccionables se entrenan cuando el usuario pulsa «Entrenar y validar», directamente en su navegador:

1. **MiniLM + regresión logística (opción inicial):** carga el encoder `Xenova/all-MiniLM-L6-v2`, que ya está preentrenado, mediante Transformers.js 3.8.1 y ONNX Runtime Web (WASM). Calcula embeddings de **384 dimensiones** de los textos de entrenamiento y validación. Entrena **desde cero, en un Web Worker**, una cabeza Softmax de tres clases usando mini-lotes y Adam. No se está entrenando desde cero el propio Transformer: únicamente el clasificador supervisado RUW.
2. **Naive Bayes o centroides TF-IDF:** construyen el vocabulario exclusivamente con el entrenamiento y ajustan sus parámetros en memoria en el navegador.
3. Se elige la postura del texto **respecto a Rusia o Ucrania**. La puntuación de validación se calcula con el conjunto oficial `Dev`, sin usar sus etiquetas para entrenar. Se muestran exactitud, **F1 macro**, matriz de confusión, métricas por clase y errores.
4. **Prueba reservada:** no se calcula su embedding ni se utiliza ninguna etiqueta hasta que se pulsa «Evaluar conjunto de prueba». Es un test independiente, nunca una métrica para seleccionar hiperparámetros.
5. **Predicción manual:** el texto introducido por el estudiante se vectoriza localmente con el mismo encoder o vectorizador; no se envía a una API de inferencia.

El Web Worker mantiene **textos y embeddings solo en memoria durante la sesión**; no persiste los vectores en GitHub ni en almacenamiento del navegador. La primera ejecución requiere descargar código de jsDelivr y archivos de modelo de Hugging Face: hay conexiones a terceros **para obtener los pesos**, no para enviar las publicaciones a inferencia remota. Si el navegador está sin conexión o no puede descargar el modelo, siguen disponibles las alternativas TF-IDF. La extracción de miles de embeddings en WASM puede tardar considerablemente en equipos poco potentes.

### Métricas históricas verificadas (referencia)

Las siguientes cifras corresponden a ejecuciones anteriores del repositorio con RUW en validación. **No son puntuaciones medidas del nuevo modelo semántico en navegador**, que deben observarse durante la ejecución.

En la auditoría con el corpus original, vocabulario de 2.500 términos y unigramas, se obtuvieron las siguientes métricas de validación **reales**, no extrapolables a opiniones de personas:

| Objetivo | Clasificador | Exactitud | F1 macro |
| --- | --- | ---: | ---: |
| Rusia | Naive Bayes | 50,8 % | 37,3 % |
| Rusia | Centroides por coseno | 50,3 % | 48,8 % |
| Ucrania | Naive Bayes | 53,6 % | 47,2 % |
| Ucrania | Centroides por coseno | 52,3 % | 51,5 % |
| Rusia | **Ridge conjunto preentrenado** | **56,8 %** | **53,9 %** |
| Ucrania | **Ridge conjunto preentrenado** | **55,6 %** | **53,2 %** |

El modelo Ridge conjunto histórico (que ya no aparece como opción para entrenar en la web) considera las combinaciones `AGAINST/FAVOR/NONE` respecto a ambos objetivos en lugar de predecir dos etiquetas de forma aislada. Utiliza TF-IDF de palabras hasta trigramas y de caracteres dentro de palabra (`char_wb`), con regularización Ridge (`alpha=1`). Fue seleccionado **solo sobre los 599 textos de validación**, por lo que las cifras mostradas no son resultados de una prueba final independiente. El alumno puede reservar los 1.187 textos de prueba para evaluarlo después.

El artefacto histórico `data/ruw-joint-ridge-v1.json` (~2,4 MB), mantenido para auditoría y no cargado por el laboratorio interactivo, contiene **únicamente vocabulario, IDF, pesos cuantizados e información de validación**, nunca las publicaciones originales. La generación es reproducible mediante `node scripts/benchmark-ruw-data.mjs` y `python3 scripts/export-ruw-joint-model.py` (requiere scikit-learn). La verificación `node scripts/verify-ruw-ridge.mjs` compara el modelo de scikit-learn con JavaScript en los mismos 599 textos: coincidencia en **597/599 predicciones conjuntas**, y exactitudes equivalentes para ambos objetivos tras cuantizar los pesos. El modelo NO es una inferencia fiable de las opiniones de una persona.

Antes de esta actualización se compararon 70 combinaciones de modelos estadísticos y representaciones de palabras y caracteres, además de modelos de embeddings semánticos preentrenados **MiniLM** y **BGE-small**, sin mejora de F1 macro sobre la mejor opción léxica en ese experimento. El nuevo clasificador semántico tampoco tiene garantizada una mejora: el estudiante debe contrastarla con la validación real. Para el modelo conjunto, los mejores valores F1 macro encontrados fueron **53,9 % en Rusia** y **53,2 % en Ucrania**; no conviene presentar los porcentajes superiores publicados en otros estudios como si fuesen los de este experimento.

Los clasificadores son deliberadamente manejables para un taller en el navegador, aunque la primera extracción con MiniLM consume más recursos. No replican los algoritmos avanzados ni las cifras del artículo RUW; ofrecen una **línea de base reproducible**. La exactitud puede parecer superior aun cuando algunas clases tengan una sensibilidad baja: por eso incluimos F1 macro y matriz de confusión.

Las pruebas automatizadas ejecutan `npm test`; para auditar la fuente RUW con conexión a internet, ejecuta `node scripts/verify-ruw.mjs`. La integración real del navegador se valida mediante Chromium en `scripts/browser-smoke.mjs`. Los datos no se envían a una API propia de Perfilado.

## Paso 4: evaluación conjunta y recomendación

La cuarta y última fase (aproximadamente **20 minutos**) se abre en [`evaluacion.html`](evaluacion.html). Parte del encargo final del responsable: considerar un sistema que pudiera identificar personas a partir de fotografías y atribuirles afinidades políticas mediante la clasificación de publicaciones. La misión del alumno es **evaluar su viabilidad técnica y jurídica**, no implantarlo.

### Una cadena sin vínculos personales reales

1. **Identificación facial:** utilizar los resultados y rechazos por umbral estudiados en el paso 2, sin trasladar automáticamente su accuracy a otro contexto.
2. **Autoría/vinculación:** comprobar si una publicación podría asociarse legítimamente a alguien; **LFW y RUW no contienen ese enlace**.
3. **Clasificación de texto:** utilizar los resultados del paso 3 distinguiendo una postura expresada en un mensaje de las opiniones de quien lo escribió.
4. **Decisión sobre una persona:** revisar cómo los errores de cada etapa, la falta de vinculación y las inferencias no justificadas podrían acumularse y causar daños.

No se vinculan ni se descargan nombres/fotografías de LFW con tuits de RUW en este paso. Los tres casos F-101, F-202 y F-303 son imaginarios y no poseen fotografías, nombres ni tuits identificables.

### Calculadora de errores hipotéticos

El alumnado puede introducir tres tasas de acierto entre 0 y 100 %: identificación facial, vinculación de autoría y clasificación de texto. Por defecto aparecen **80 %, 70 % y 53 %**, valores **ilustrativos** que deben reemplazarse por resultados observados en los ejercicios cuando sea posible. Como no existe un dataset conjunto, **la tasa de vinculación siempre es supuesta**. La calculadora muestra:

- El producto de las tasas, únicamente como proporción de tres aciertos simultáneos **suponiendo independencia**.
- El complemento de ese producto, como frecuencia hipotética con al menos un fallo.
- Los límites de Fréchet sin asumir independencia: `máx(0, p1 + p2 + p3 - 2)` hasta `mín(p1, p2, p3)`.

**No son métricas empíricas del sistema conjunto ni miden la certeza de una opinión personal.** Los resultados reales de los pasos 2 y 3 provienen de datasets, poblaciones y tareas diferentes. El simulador no dispone de etiquetas conjuntas; no genera puntuaciones reales ni ejecuta reconocimiento sobre personas.

### Valoración jurídica y preguntas

Se plantean **tres marcos de análisis** y enlaces normativos: RGPD / LO 3/2018 en el ámbito civil, Directiva (UE) 2016/680 / LO 7/2021 para tratamientos policiales en su ámbito, y el régimen específico según finalidad y competencia para actividades exclusivamente militares o de defensa. También se señala el alcance y las limitaciones del Reglamento europeo de IA. No se presenta una declaración automática de licitud o ilicitud: debe analizarse cada supuesto concreto.

Las preguntas **20 a 23** son las previstas en la propuesta docente: diferencia entre identificación, autoría e inferencia; acumulación de errores; protección de datos y normativa por finalidad; e **informe final razonado** para el responsable. Las respuestas se guardan y exportan con el mismo módulo `src/answers.js` usado en los pasos 1–3. No hay backend, llamadas a inferencia ni archivos de opinión política vinculados a identidades reales.

## GitHub Pages

En **Settings → Pages → Build and deployment**, selecciona **Deploy from a branch → main → / (root)**. Una vez habilitado el despliegue:

https://antoniorani.github.io/Perfilado/

Las rutas relativas funcionan bajo el subdirectorio `/Perfilado/`.

**Uso exclusivamente docente.** La aplicación contiene modelos reales de NLP y reconocimiento facial, pero no es un sistema de vigilancia ni está validada para tomar decisiones sobre personas. Las publicaciones RUW no se asocian a identidades reales de LFW; la evaluación conjunta de riesgos se planteará con entidades ficticias.
