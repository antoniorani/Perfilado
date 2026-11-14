# PERFILADO

Laboratorio docente de inteligencia artificial, identificación facial y protección de datos. **Autor: Antonio Ramos Nieto.**

## Estado de la práctica

- **Introducción:** encargo del equipo técnico y recorrido de la práctica, presentado al final.
- **Paso 1:** una imagen por identidad, nombre, código de perfil y atributos sintéticos. Exploración con búsqueda, filtros, gráficos y preguntas de reflexión.
- **Paso 2:** identificación facial real en navegador: embeddings de 128 dimensiones, comparación 1:N, umbral y evaluación con personas registradas y desconocidas.
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
src/biometric-core.js      Muestra LFW y evaluación de distancias
src/face-engine.js         Modelos preentrenados reales
src/step2.js               UI interactiva del experimento
tests/biometrics.test.js   Pruebas de identificación
data/lfw-identities.json   Índice real LFW
tests/core.test.js         Pruebas automatizadas
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

## Paso 2: reconocimiento facial real

La página `biometria.html` carga **face-api.js 0.22.2** desde jsDelivr y, únicamente al comenzar el experimento, descarga tres redes preentrenadas: detector Tiny Face Detector, detector de 68 puntos faciales y red de reconocimiento de 128 componentes. Para comparar caras se calcula la **distancia euclidiana** entre embeddings; el umbral de 0,60 es un valor didáctico ajustable, **no una probabilidad ni una garantía de identidad**.

- El paso 1 mantiene los **5.749 perfiles originales**, pero calcular embeddings en directo de 13.233 imágenes no resulta viable para una clase con navegadores corrientes. El paso 2 utiliza un **subconjunto reproducible de 4, 8 o 12 identidades registradas** y elige por cada una una referencia y una segunda foto. Incluye además identidades que no están en la galería para comprobar falsos positivos y rechazos.
- Todas las distancias visibles se obtienen de **embeddings reales calculados por el navegador**. No se han incluido puntuaciones sintéticas en la interfaz. Si no se detecta un rostro o falla la carga remota, el ejemplo se excluye y se informa de cuántos casos se han perdido.
- Las fotografías se sirven desde Hugging Face y los pesos del modelo desde jsDelivr. **No se suben imágenes ni embeddings generados por el usuario**, pero las peticiones de descarga se realizan a esos proveedores externos. Es necesaria una conexión estable y que Hugging Face permita cargar las imágenes con CORS para la lectura de píxeles.
- La precisión que muestra la página es **aciertos entre asignaciones aceptadas** y es distinta de la exactitud global. Los tipos de resultados son: identificación correcta, identidad equivocada, conocido no identificado, desconocido rechazado y falso aviso sobre un desconocido.
- El modelo fue desarrollado y evaluado públicamente sobre conjuntos de reconocimiento facial, incluido LFW; **esta práctica no es una evaluación independiente**, y una muestra tan reducida no es representativa de un despliegue. No utilices las cifras para justificar decisiones sobre personas.
- No se asocian perfiles políticos o textos de RUW a las personas reales fotografiadas en LFW.

### Próximas extensiones

Si fuese necesario cubrir los 5.749 perfiles y las 7.484 fotografías adicionales, habría que **precalcular embeddings reproducibles en un entorno de preparación** y publicar solo los artefactos compatibles con los derechos aplicables, separados de los datos biográficos y políticos. No presentar el experimento de navegador como una evaluación completa del dataset.

## GitHub Pages

En **Settings → Pages → Build and deployment**, selecciona **Deploy from a branch → main → / (root)**. Una vez habilitado el despliegue:

https://antoniorani.github.io/Perfilado/

Las rutas relativas funcionan bajo el subdirectorio `/Perfilado/`.

**Uso exclusivamente docente.** La app no opera un sistema de vigilancia y no ejecuta aún modelos de procesamiento del lenguaje. La etapa conjunta sobre opiniones políticas deberá construirse con identidades totalmente ficticias.
