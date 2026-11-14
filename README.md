# PERFILADO

Laboratorio docente de inteligencia artificial, identificación facial y protección de datos. **Autor: Antonio Ramos Nieto.**

## Primera fase disponible

- **Introducción:** encargo del equipo técnico y recorrido de la práctica, presentado al final.
- **Paso 1:** una imagen por identidad, nombre, código de perfil y atributos sintéticos. Exploración con búsqueda, filtros, gráficos y preguntas de reflexión.
- **Pasos 2–4:** identificación facial, análisis NLP y evaluación conjunta. Pendientes de desarrollar con modelos reales.

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

## GitHub Pages

En **Settings → Pages → Build and deployment**, selecciona **Deploy from a branch → main → / (root)**. Una vez habilitado el despliegue:

https://antoniorani.github.io/Perfilado/

Las rutas relativas funcionan bajo el subdirectorio `/Perfilado/`.

**Uso exclusivamente docente.** La app no opera un sistema de vigilancia ni ejecuta aún modelos faciales o de procesamiento del lenguaje. La etapa conjunta sobre opiniones políticas deberá construirse con identidades totalmente ficticias.
