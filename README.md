# PERFILADO

Laboratorio docente de inteligencia artificial, identificación biométrica y protección de datos.

## Alcance de esta primera entrega

- **Introducción**: presentación narrativa del encargo y cuatro etapas del taller.
- **Paso 1**: catálogo exploratorio con 5.749 perfiles, búsqueda, ordenación, filtros, gráficos, exportación CSV y cinco preguntas de reflexión.
- **Fotografías**: carga local opcional en el navegador. No se redistribuyen en GitHub.
- **Próximamente**: identificación con modelos faciales reales, NLP con modelos textuales reales y evaluación conjunta.

### Datos y procedencia

El [dataset LFW](https://vis-www.cs.umass.edu/lfw/) (Huang et al., 2007) contiene 13.233 fotografías correspondientes a 5.749 identidades. El archivo `data/lfw-identities.json` incluye **únicamente los nombres originales de las identidades y su número de fotografías**, tomados de [lfw-names.txt](https://github.com/happynear/FaceDatasets/blob/master/CASIA/lfw-names.txt) y contrastados con los totales oficiales.

Los perfiles de clase `PF-00001` a `PF-05749` tienen una sola imagen de referencia (`_0001.jpg`). **Edad, zona e idioma son generados mediante un algoritmo determinista y son enteramente ficticios**; no describen a las personas reales fotografiadas. No se asigna ninguna opinión política a identidades de LFW.

LFW incluye fotografías de personas reales obtenidas en Internet; que un dataset sea accesible públicamente no equivale a una autorización general para redistribuir las imágenes o destinarlas a usos diferentes. Por ello **este repositorio no contiene fotografías de LFW**. El docente puede seleccionar desde su ordenador una carpeta o un grupo de archivos JPG descargados lícitamente. El navegador indexa el nombre de archivo (`Nombre_Apellido_0001.jpg`) y muestra las imágenes que encuentre: no se sube nada a ningún servidor.

Las respuestas de los estudiantes se conservan solamente en su navegador mediante `localStorage` cuando está disponible; pueden exportarse a un archivo TXT. El CSV exportado contiene el perfil y sus variables ficticias, incluida la referencia al archivo de imagen, no la imagen.

## Implementación

HTML, CSS y **módulos ES de JavaScript**, sin framework, dependencias externas ni proceso de compilación. Preparado para funcionar en GitHub Pages, también bajo el subdirectorio `/Perfilado/`.

```text
index.html                 # Introducción
datos.html                 # Primer paso
assets/styles.css          # Diseño responsive
src/core.js                # Transformaciones puras y comprobables
src/step1.js               # Comportamiento de la interfaz
data/lfw-identities.json   # Catálogo original sin fotografías
tests/core.test.js         # Pruebas de integridad y funciones
```

### Ejecutar localmente

```bash
python3 -m http.server 8000
# abrir http://localhost:8000
```

No abras los archivos mediante `file://`: la carga del catálogo mediante `fetch()` requiere HTTP. Para ejecutar las pruebas con Node.js (versión actual):

```bash
npm test
```

No es necesario ejecutar `npm install`.

### Publicar con GitHub Pages

Después de fusionar la rama de trabajo con `main`:

1. Abre **Settings → Pages** en el repositorio.
2. En **Build and deployment**, selecciona **Deploy from a branch**.
3. Elige `main` y `/ (root)`; guarda los cambios.

La dirección prevista será `https://antoniorani.github.io/Perfilado/`. La publicación efectiva depende de que GitHub Pages esté habilitado.

## Próximas fases

El paso 2 incorporará embeddings faciales obtenidos mediante un **modelo preentrenado real**, comparación 1:N y errores de identificación. El paso 3 añadirá clasificación de posturas **expresadas en textos**, sin atribuir opiniones ficticias a personas reales. El paso 4 se realizará con casos e identidades completamente ficticias y revisará los riesgos de combinar ambos sistemas.

**Uso exclusivo para formación.** No es un sistema de identificación operacional y no se deben interpretar las puntuaciones de modelos como pruebas de identidad, ideología o conducta de una persona.
