# PERFILADO

Laboratorio docente de inteligencia artificial, identificación facial y protección de datos. **Autor: Antonio Ramos Nieto.**

## Estado de la práctica

- **Introducción:** encargo del equipo técnico y recorrido de la práctica, presentado al final.
- **Paso 1:** una imagen por identidad, nombre, código de perfil y atributos sintéticos. Exploración con búsqueda, filtros, gráficos y preguntas de reflexión.
- **Paso 2:** 5.749 embeddings iniciales precalculados y 100 nuevas imágenes de LFW procesadas en el navegador, con búsqueda 1:N, top-1 y ajuste de umbral.
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
src/step2.js               Experimento real de 100 fotografías
src/full-gallery.js        Selección reproducible, lectura f32 y búsqueda top-5
scripts/build-gallery.mjs   Generación reproducible de embeddings LFW
scripts/combine-gallery.mjs Ensamblado y validación de 16 lotes
.github/workflows/precompute-gallery.yml  Proceso de precálculo
data/lfw-identities.json   Índice real LFW
tests/core.test.js         Pruebas del catálogo
 tests/full-gallery.test.js  Pruebas de búsqueda y umbrales
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

## Paso 2: 5.749 referencias / 100 fotografías de prueba

### Método y comportamiento

La nueva práctica separa el cálculo de **5.749 embeddings de referencia** de la búsqueda en tiempo real de **100 fotografías nuevas**:

1. El proceso de preparación usa la **primera foto de cada una de las 5.749 identidades** de LFW y calcula un vector de 128 componentes por persona con `face-api.js@0.22.2`. Se distribuye en 16 lotes de GitHub Actions y se ensambla en un único archivo binario `lfw-gallery.f32` de **2.943.488 bytes** (5.749 × 128 × 4).
2. El alumnado recibe 100 fotografías de prueba, una segunda foto por cada una de **100 identidades diferentes y conocidas**, seleccionadas de forma determinista entre las 1.680 personas que tienen varias fotos en LFW. No se reutiliza la misma imagen inicial.
3. Al iniciar el ejercicio se descargan los pesos preentrenados y el navegador calcula **un embedding real por fotografía de prueba** con el mismo método: `faceapi.computeFaceDescriptor(image)` sobre la imagen LFW original, de 250×250 píxeles.
4. Cada vector de prueba se compara con **todas las referencias válidas** mediante distancia euclidiana y se muestran las cinco coincidencias más próximas. La distancia no es una probabilidad.
5. **Top-1** significa que el candidato más cercano es la identidad correcta *sin umbral*. Un umbral configurable determina si la coincidencia se **acepta** o **rechaza**. El alumno observa aciertos aceptados, identificaciones erróneas aceptadas y rechazos. Si hay fallos de imagen o del modelo, se cuentan como **no procesadas**, no como predicciones.

**Límite experimental:** Las 100 personas de prueba figuran en la galería. Este experimento **no mide la tasa de identificación errónea de personas desconocidas**. El modelo de reconocimiento facial se ha utilizado previamente para estudiar LFW, de modo que el resultado no es una validación independiente ni permite extrapolaciones a una población real.

### Generación del archivo de referencias

La automatización `Precompute real LFW gallery (evaluation)` puede iniciarse manualmente desde **Actions**. Sus 16 trabajos calculan los embeddings reales con la misma versión de la biblioteca y después comprueban el orden, el formato y la cantidad de vectores. La acción produce un artefacto descargable con:

- `lfw-gallery.f32`: matriz de float32 de 128 dimensiones por identidad, ordenada igual que `data/lfw-identities.json`.
- `lfw-gallery-report.json`: trazabilidad del modelo, recuentos, fallos, índices y tiempos de cada lote.

No se crean embeddings ficticios cuando una imagen falla: se representa como no válida y se documenta en el informe. La validación de integridad exige 5.749 posiciones y señala si alguna referencia no pudo calcularse.

### Decisión de publicación y privacidad

**Los embeddings faciales no son datos anónimos**: son representaciones biométricas calculadas a partir de fotografías de personas identificables. Por diseño, **el código no obliga a incorporarlos a un repositorio público**. El docente puede descargar el artefacto desde GitHub Actions y cargar el archivo `lfw-gallery.f32` desde la web, sin transmitir ese archivo al servidor.

La página intentará cargar automáticamente `data/lfw-gallery.f32` **si el responsable del proyecto decide incorporarlo tras revisar la licitud y las condiciones de difusión**. Si no existe, la interfaz informa de ello y permite seleccionar el archivo localmente.

La publicación abierta del conjunto completo de plantillas **exige evaluación específica de protección de datos**; su presencia en un repositorio público no debe decidirse por mera conveniencia técnica.

Las peticiones de fotografías de LFW y de pesos del modelo se realizan a proveedores externos; los embeddings de las nuevas fotos se calculan en el navegador y no se envían a un backend. Las respuestas del alumnado permanecen en `localStorage` o en los archivos TXT que exporten.

## GitHub Pages

En **Settings → Pages → Build and deployment**, selecciona **Deploy from a branch → main → / (root)**. Una vez habilitado el despliegue:

https://antoniorani.github.io/Perfilado/

Las rutas relativas funcionan bajo el subdirectorio `/Perfilado/`.

**Uso exclusivamente docente.** La app no opera un sistema de vigilancia y no ejecuta aún modelos de procesamiento del lenguaje. La etapa conjunta sobre opiniones políticas deberá construirse con identidades totalmente ficticias.
