# Chatbot conversacional con procesamiento del lenguaje natural

Agente en **Python** que responde preguntas en **español** usando técnicas clásicas de **PLN** (procesamiento del lenguaje natural): reglas, gramáticas formales, análisis morfosintáctico, léxico y recuperación de información. Puedes usarlo en **consola** con entrada de texto o voz, o como **bot de Telegram**.

El código está organizado en módulos pequeños para que se vea **qué hace cada pieza** (intención, sintaxis, tema, respuesta) en lugar de ocultarlo detrás de un único modelo generativo.

---

## ¿Qué es y para qué sirve?

Es un **chatbot por turnos**: el usuario escribe (o envía mensajes en Telegram) y el sistema devuelve una respuesta basada en:

- Una **base de conocimiento** en JSON (definiciones y textos que tú defines).
- **WordNet** para sinónimos y definiciones léxicas.
- Opcionalmente un índice **RAG** local (búsqueda por similitud de texto con TF-IDF) cuando la base JSON y WordNet no alcanzan.

Sirve para:

- Tener un asistente temático (por defecto: computación e IA) que puedes **personalizar** cambiando el JSON.
- Probar en un solo proyecto **gramáticas**, **etiquetado**, **morfología** y **agentes conversacionales** con librerías habituales (NLTK, spaCy).
- Aprender o mostrar un pipeline de PLN **transparente** (con trazas de depuración en consola si lo deseas).

---

## Cómo funciona (visión general)

En cada mensaje el flujo es **entender → buscar respuesta → responder**:

1. **Entrada** — texto del usuario (consola o Telegram) o voz capturada en consola con `/voz`.
2. **Intención** — patrones frecuentes (saludo, despedida, «qué es…») se detectan con una [gramática de rasgos](https://www.nltk.org/howto/featstruct.html) en [NLTK](https://www.nltk.org/) (`intents.py`).
3. **Sintaxis (opcional)** — la frase se resume a una secuencia de categorías y se valida con una [gramática libre de contexto (CFG)](https://en.wikipedia.org/wiki/Context-free_grammar) y un analizador tipo [Chart Parser](https://www.nltk.org/book/ch08.html) (`syntax_cfg.py`). Sirve para inspeccionar la estructura; el bot puede seguir aunque el análisis falle.
4. **Tema** — en preguntas definicionales, [spaCy](https://spacy.io/) ayuda a extraer el concepto buscado (`extraction.py`); los acrónimos (IA, ML, DL, etc.) se normalizan (`nlp_utils.py`).
5. **Respuesta** — en este orden: entrada en el **JSON** → **WordNet** → coincidencia parcial por palabras → **RAG** (si hay índice generado y está activado).

En **consola** puedes activar líneas `DEBUG …` para ver el recorrido. En **Telegram** la conversación va sin ese ruido técnico.

### Recorrido del código (para explicar el proyecto)

Cada archivo `.py` del paquete `pln_chatbot/` lleva al inicio un comentario en lenguaje llano: qué hace ese módulo y por qué está ahí. Orden sugerido al presentarlo:

1. **`cli.py`** / **`telegram_bot.py`** — dónde arranca el programa (consola o Telegram).
2. **`interaction.py`** — un mensaje entra; se reparten comandos o se manda al diálogo.
3. **`dialogue.py`** — intención (NLTK → spaCy) y respuesta (JSON → WordNet → RAG).
4. **`intents.py`**, **`extraction.py`**, **`syntax_cfg.py`**, **`morphology.py`**, **`wordnet_tools.py`** — piezas de PLN que puedes enseñar por separado.
5. **`knowledge.py`**, **`rag.py`**, **`rag_ingest.py`** — de dónde salen las respuestas y cómo se arma el índice.
6. **`resources.py`**, **`config.py`** — carga de modelos y rutas.

El script **`morfosintactico.py`** (si lo tienes en local) es aparte: demo de morfosintaxis, no lo usa el chatbot al correr.

---

## Tecnologías y piezas del proyecto

### NLTK — intenciones, sintaxis y WordNet

- **Intenciones**: `FeatureGrammar` + `FeatureChartParser` para frases cortas y predecibles (saludos, preguntas tipo «qué es X»). Documentación: [NLTK — parsing con rasgos](https://www.nltk.org/howto/featstruct.html).
- **Sintaxis**: CFG propia en `syntax_cfg.py`; concepto: [gramática libre de contexto](https://en.wikipedia.org/wiki/Context-free_grammar), libro NLTK [cap. 8 — Análisis sintáctico](https://www.nltk.org/book/ch08.html).
- **WordNet**: red léxica para definiciones y sinónimos; en este proyecto se usa el español vía `lang='spa'`. Referencia: [WordNet (Princeton)](https://wordnet.princeton.edu/), acceso en NLTK: [corpus WordNet](https://www.nltk.org/howto/wordnet.html).

### spaCy — morfosintaxis y extracción de tema

- Modelo por defecto: **`es_core_news_lg`** (etiquetas POS, lemas, *noun chunks*).
- Comando de instalación del modelo (una vez por entorno): `python -m spacy download es_core_news_lg`.
- Documentación: [spaCy — modelos en español](https://spacy.io/models/es).

### Base de conocimiento (JSON)

- Archivo declarativo con `entries` (clave → definición). Por defecto: `data/knowledge.default.json`.
- Hay un ejemplo de otro dominio en `data/knowledge.ventas.example.json` para copiar y adaptar.

### RAG local (opcional)

- **RAG** aquí = **recuperar** fragmentos relevantes y mostrarlos al usuario (no hay modelo generativo grande; es recuperación + texto citado).
- Índice **TF-IDF** con [scikit-learn](https://scikit-learn.org/), guardado en `data/rag/rag_tech_index.joblib`.
- Cómo se **descarga**, **filtra** y **construye** el índice: sección [Índice RAG paso a paso](#índice-rag-paso-a-paso).
- El `.joblib` **no** va en Git (se genera en tu máquina).

### Telegram

- [python-telegram-bot](https://docs.python-telegram-bot.org/) en modo *long polling*: el proceso debe estar en ejecución en tu máquina mientras usas el bot.
- Token con [@BotFather](https://t.me/BotFather); no lo subas a Git.

### Entrada por voz

- [SpeechRecognition](https://pypi.org/project/SpeechRecognition/) captura audio desde el micrófono y lo convierte a texto.
- En consola se usa con el comando `/voz`; el texto reconocido pasa por el mismo pipeline de NLTK, spaCy, WordNet y base JSON.
- En macOS puede requerir permisos de micrófono para Terminal o el IDE, y soporte de audio con `PyAudio`.

---

## Requisitos

- **Python**: 3.10 o superior recomendado.
- **Sistema**: probado en **Windows** (PowerShell); los comandos de activación de venv se dan para Windows; en Linux/macOS cambia la ruta de activación (ver [documentación de venv](https://docs.python.org/3/library/venv.html)).
- **Red**: necesaria la primera vez para descargar modelos NLTK/spaCy y, si usas RAG, para `rag_ingest`.
- **Micrófono** (opcional): necesario para usar `/voz` en consola.
- **Telegram** (opcional): cuenta y token de bot.

---

## Entorno virtual e instalación

Aislar dependencias en `.venv` evita conflictos con otros proyectos.

### 1. Crear el entorno (Windows, PowerShell)

Desde la carpeta raíz del repositorio (donde está `requirements.txt`):

```powershell
python -m venv .venv
```

### 2. Activar el entorno

**PowerShell** (si falla por política de ejecución, una vez: `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned`):

```powershell
.\.venv\Scripts\Activate.ps1
```

**Símbolo del sistema (cmd)**:

```bat
.\.venv\Scripts\activate.bat
```

El prompt suele mostrar `(.venv)` al inicio.

### 3. Instalar dependencias

Con el entorno **activado**:

```powershell
python -m pip install --upgrade pip
pip install -r requirements.txt
```

Eso instala lo fijado en `requirements.txt`: `nltk`, `spacy`, `python-telegram-bot`, `datasets`, `scikit-learn`, `SpeechRecognition`, `PyAudio` (y dependencias como `joblib`).

### 4. Modelo de spaCy (obligatorio)

No viene dentro de `requirements.txt`; instálalo aparte:

```powershell
python -m spacy download es_core_news_lg
```

### 5. Recursos NLTK

Al primer arranque el proyecto intenta descargar `punkt`, etiquetador, `wordnet`, `omw-1.4`, etc. Si falla por red o permisos:

```python
import nltk
nltk.download("punkt")
nltk.download("wordnet")
nltk.download("omw-1.4")
nltk.download("punkt_tab")
nltk.download("averaged_perceptron_tagger")
```

### 6. Índice RAG (opcional)

Si quieres el respaldo RAG, después de instalar dependencias ejecuta una vez:

```powershell
python -m pln_chatbot.rag_ingest
```

Detalle completo (de dónde se descarga cada corpus y qué hace el script): [Índice RAG paso a paso](#índice-rag-paso-a-paso). Sin índice el chatbot **arranca igual**; solo no usará RAG (o `PLN_CHATBOT_USE_RAG=0`).

---

## Índice RAG paso a paso

Esta sección describe **cómo se arma el RAG de este proyecto**: no es un servicio en la nube; es un índice **local** que tú generas con `pln_chatbot/rag_ingest.py` y que el chatbot consulta en `pln_chatbot/rag.py`.

### Qué hace el RAG en este chatbot

1. El usuario pregunta algo que **no** está en el JSON ni en WordNet (o la coincidencia es débil).
2. Si existe `data/rag/rag_tech_index.joblib` y `PLN_CHATBOT_USE_RAG=1`, se buscan los **fragmentos más parecidos** a la pregunta (similitud coseno sobre vectores TF-IDF).
3. Se devuelve un texto basado en esos fragmentos, indicando la **fuente** (Wikipedia en español o AbScientia).

No entrena un LLM: solo **recupera** trozos de texto ya ingeridos.

### De dónde se descargan los datos

Todo se obtiene en **streaming** con la librería [`datasets`](https://huggingface.co/docs/datasets) de **Hugging Face** (hace falta **Internet** la primera vez que corres `rag_ingest`).

| Origen | Repositorio en Hugging Face | Contenido |
|--------|-----------------------------|-----------|
| Wikipedia en español | [spanish-ir/eswiki_20240401_corpus](https://huggingface.co/datasets/spanish-ir/eswiki_20240401_corpus) | Artículos (campos típicos: `title`, `text`). |
| Resúmenes científicos (STEM) | [BSC-LT/AbScientia](https://huggingface.co/datasets/BSC-LT/AbScientia) | Textos tipo abstract (el script prueba columnas `abstract`, `text`, `sentence`, etc.). |

Revisa en cada página de Hugging Face la **licencia** y condiciones de uso si vas a redistribuir el índice o el corpus.

### Pasos que ejecuta `rag_ingest` (resumen del código)

| Paso | Qué ocurre | Archivo |
|------|------------|---------|
| 1 | Conecta a cada dataset en modo **streaming** (no descarga todo el disco de una vez). | `rag_ingest.py` → `_load_streaming_dataset` |
| 2 | **Limpia** espacios y saltos de línea. | `_clean_text` |
| 3 | **Filtra** párrafos que contengan palabras clave de tecnología/IA (misma lógica que las consultas tech en `rag.py`). | `has_tech_keyword` en `rag.py` |
| 4 | **Parte** el texto en fragmentos (~700 caracteres, solapamiento ~120, cortando preferentemente en `. `). | `_chunk_text` |
| 5 | Repite hasta alcanzar los límites por fuente (por defecto **25 000** chunks de eswiki y **12 000** de AbScientia). | `--max-eswiki`, `--max-abscientia` |
| 6 | Construye matriz **TF-IDF** (unigramas y bigramas, hasta 80 000 características, `min_df=2`). | `build_index` en `rag.py` |
| 7 | Guarda vectorizador + matriz + lista de documentos en **`data/rag/rag_tech_index.joblib`**. | `save_index` |

### Comandos para generar el índice

Desde la raíz del repo, venv activado y `pip install -r requirements.txt` ya hecho:

```powershell
# Valores por defecto del script (25k + 12k chunks)
python -m pln_chatbot.rag_ingest
```

Menos datos (más rápido, menos RAM):

```powershell
python -m pln_chatbot.rag_ingest --max-eswiki 5000 --max-abscientia 3000
```

Salida esperada en log: progreso por dataset, número de chunks y mensaje final con la ruta del `.joblib`.

### Usar el índice en el chatbot

- Ruta por defecto: `data/rag/rag_tech_index.joblib` (definida en `pln_chatbot/config.py` como `RAG_INDEX_PATH`).
- Otra ruta: variable `PLN_CHATBOT_RAG_INDEX`.
- Desactivar RAG sin borrar el archivo: `PLN_CHATBOT_USE_RAG=0`.

Al arrancar `python -m pln_chatbot`, `resources.py` intenta **cargar** el índice; si no existe, el bot funciona con JSON y WordNet y el log lo indica.

### Regenerar o cambiar el dominio

- Si cambias filtros en `rag.py` / `rag_ingest.py` o quieres corpus distintos, vuelve a ejecutar `rag_ingest` (sobrescribe el `.joblib`).
- El índice debe generarse con **las mismas versiones** de `scikit-learn` que en `requirements.txt` para evitar incompatibilidades al cargar.

---

## Guía de ejecución (resumen)

| Modo | Para qué sirve | Requisitos extra |
|------|----------------|------------------|
| **Consola** | Probar el motor en tu PC, ver trazas `DEBUG`, comandos `/trace`. | Solo instalación del proyecto (sección [Entorno virtual e instalación](#entorno-virtual-e-instalación)). |
| **Telegram** | Hablar con el bot desde el móvil o Telegram Desktop. | Cuenta en [Telegram](https://telegram.org/) + bot en [@BotFather](https://t.me/BotFather) + token + PC encendida con el script en ejecución. |

En ambos casos: carpeta **raíz** del repositorio, entorno virtual **activado** y dependencias ya instaladas.

---

## Ejecutar en consola (PowerShell o CMD)

### Antes de empezar (una sola vez)

1. Clona o descarga el repositorio y entra en la carpeta (donde está `requirements.txt`):

```powershell
cd "ruta\al\proyecto\Chatbot"
```

2. Crea y activa el entorno virtual (ver [Entorno virtual e instalación](#entorno-virtual-e-instalación)): `python -m venv .venv` → activar → `pip install -r requirements.txt` → `python -m spacy download es_core_news_lg`.

3. (Opcional) Genera el índice RAG si quieres respuestas de respaldo: [Índice RAG paso a paso](#índice-rag-paso-a-paso).

### Cada vez que quieras usar el chatbot en consola

**PowerShell** (recomendado en Windows):

```powershell
cd "ruta\al\proyecto\Chatbot"
.\.venv\Scripts\Activate.ps1
python -m pln_chatbot
```

**Símbolo del sistema (cmd)**:

```bat
cd ruta\al\proyecto\Chatbot
.\.venv\Scripts\activate.bat
python -m pln_chatbot
```

**Atajo** (misma función que el comando anterior):

```powershell
python chatbot.py
```

### Qué verás en pantalla

- Mensaje de bienvenida y dominio cargado desde `data/knowledge.default.json`.
- Prompt para escribir (consola interactiva).
- Prueba: `hola`, `qué es machine learning`, `/ayuda`, `/temas`, `/morfologia el algoritmo es eficiente`, `/wordnet algoritmo`.
- Entrada por voz: escribe `/voz`, habla cuando aparezca `Escuchando... hable ahora.`, y el texto reconocido se procesa como una pregunta normal.
- Para salir: `salir`, `/salir` o `Ctrl+C`.

### Nota para entrada por voz en macOS

Si `PyAudio` no instala correctamente, instala primero `portaudio` y vuelve a instalar dependencias:

```bash
brew install portaudio
pip install PyAudio
```

macOS puede pedir permiso de micrófono para Terminal, iTerm o el IDE desde donde ejecutes el programa.

### Trazas de depuración (solo consola)

```text
/trace on
/trace off
```

O antes de arrancar:

```powershell
$env:PLN_CHATBOT_TRACE="0"
python -m pln_chatbot
```

### Usar otro archivo de conocimiento (opcional)

```powershell
$env:PLN_CHATBOT_KNOWLEDGE="data\knowledge.ventas.example.json"
python -m pln_chatbot
```

---

## Ejecutar en Telegram (crear tu bot y conectarlo)

El bot usa **long polling**: tu computadora ejecuta `telegram_bot.py`, consulta a los servidores de Telegram y responde. **No es un servidor en la nube**: si cierras PowerShell o apagas el PC, el bot deja de contestar.

Solo responde en chats **privados** con el bot (no en grupos).

### Paso 1 — Instalar el proyecto en tu PC

Igual que para consola: venv activado, `pip install -r requirements.txt`, modelo spaCy y (opcional) índice RAG.

### Paso 2 — Crear el bot en Telegram (BotFather)

1. Abre Telegram (móvil o escritorio) y busca **[@BotFather](https://t.me/BotFather)** (cuenta oficial con marca de verificación).
2. Pulsa **Iniciar** o envía `/start`.
3. Envía **`/newbot`**.
4. BotFather pide un **nombre visible** (ej.: `Asistente PLN Demo`). Es el nombre que verán los usuarios en la lista de chats.
5. Pide un **usuario** del bot; debe terminar en `bot` (ej.: `mi_asistente_pln_bot`). Si el nombre ya está ocupado, prueba otro.
6. Si todo va bien, BotFather envía el **token de API**, con forma similar a:

   `7123456789:AAHxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx`

7. **Copia el token** y guárdalo en un lugar seguro. **No lo subas a GitHub**, foros ni capturas públicas. Quien tenga el token puede controlar tu bot.

**Comandos útiles de BotFather (opcionales):**

| Comando | Para qué |
|---------|----------|
| `/mybots` | Administrar tus bots (nombre, foto, descripción). |
| `/setdescription` | Texto que ve el usuario antes de pulsar *Iniciar*. |
| `/setabouttext` | Texto breve en el perfil del bot. |
| `/revoke` | Invalidar el token y generar uno nuevo (si filtraste el token). |

Más información: [Introducción a bots de Telegram](https://core.telegram.org/bots).

### Paso 3 — Configurar el token en tu PC

El programa lee la variable **`TELEGRAM_BOT_TOKEN`**. Defínela en la **misma sesión** donde ejecutes el bot (o en un `.env` local que no subas a Git).

**PowerShell** (sustituye por tu token real):

```powershell
cd "ruta\al\proyecto\Chatbot"
.\.venv\Scripts\Activate.ps1
$env:TELEGRAM_BOT_TOKEN="7123456789:AAHxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx"
python -m pln_chatbot.telegram_bot
```

**CMD**:

```bat
cd ruta\al\proyecto\Chatbot
.\.venv\Scripts\activate.bat
set TELEGRAM_BOT_TOKEN=7123456789:AAHxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
python -m pln_chatbot.telegram_bot
```

**Archivo `.env` (opcional):** una línea `TELEGRAM_BOT_TOKEN=...` en la raíz, si tu herramienta carga variables desde ahí. El archivo está en `.gitignore`.

Sin token, el programa imprime un error con instrucciones y **no** arranca.

### Paso 4 — Comprobar que el proceso está en marcha

En la terminal deberían aparecer mensajes de inicio (carga del JSON, spaCy, etc.). **Deja esa ventana abierta**.

Al iniciar, se registra el menú de comandos en Telegram: `/start`, `/ayuda`, `/temas`, `/morfologia`, `/syntax`, `/wordnet`, `/salir` (visibles al pulsar `/` en el chat).

### Paso 5 — Probar el bot desde Telegram

1. Busca en Telegram el **usuario** de tu bot (ej. `@mi_asistente_pln_bot`).
2. Abre chat **privado** (no grupo).
3. Pulsa **Iniciar** o envía `/start` (mensaje de bienvenida).
4. Prueba `/ayuda`, `/temas`, `qué es inteligencia artificial`, `/morfologia el sistema procesa datos`, `/wordnet red`.

Si no responde: token correcto, script en ejecución en la PC, chat privado.

### Paso 6 — Detener el bot

`Ctrl+C` en la terminal donde corre `telegram_bot.py`. Vuelve a ejecutar el Paso 3 para ponerlo otra vez en línea.

### Configuración adicional (Telegram + proyecto)

| Qué cambiar | Dónde |
|-------------|--------|
| Conceptos y respuestas | `data/knowledge.default.json` o `PLN_CHATBOT_KNOWLEDGE` |
| Desactivar RAG | `PLN_CHATBOT_USE_RAG=0` antes de arrancar |
| Texto de bienvenida `/start` | `TELEGRAM_WELCOME` en `pln_chatbot/telegram_bot.py` |
| Comandos del menú `/` | función `post_init` en `pln_chatbot/telegram_bot.py` |

`/trace` y `/debug` **no** están en Telegram; úsalos en consola para ver el pipeline interno.

### Limitaciones (antes de una demo pública)

- Sin un servidor desplegado, el bot **no** está 24/7 en internet.
- No ejecutes **dos** instancias con el mismo token a la vez.
- La primera respuesta puede tardar unos segundos (carga de NLTK, spaCy y RAG opcional).

---

## Comandos disponibles

| Comando | Consola | Telegram |
|--------|---------|----------|
| `/ayuda` | Ayuda completa (incluye `/trace`, `/debug`) | Ayuda resumida + menú `/` |
| `/temas` | Lista temas del JSON | Igual |
| `/morfologia <frase>` | Análisis morfológico (spaCy) | Igual (formato adaptado) |
| `/syntax <frase>` | Árbol / validación CFG | Igual (puede truncarse si es muy largo) |
| `/wordnet <término>` | Definiciones y sinónimos WordNet | Igual |
| `/salir` | Cierra la sesión de consola | Mensaje de despedida |
| `/trace`, `/debug` | Trazas `DEBUG` en la terminal | No disponibles |

También puedes escribir en lenguaje natural («qué es machine learning», «hola») o usar `wordnet define <término>` y `salir` sin barra en consola.

---

## Configuración

### Archivo `pln_chatbot/config.py`

Centraliza rutas por defecto:

- `KNOWLEDGE_PATH` — JSON de dominio.
- `RAG_INDEX_PATH` — archivo `.joblib` del índice RAG.
- `SPACY_MODEL` — nombre del pipeline spaCy.
- `USE_RAG_FALLBACK` — si se intenta RAG cuando no hay hito en JSON/WordNet.

### Variables de entorno (opcional)

| Variable | Efecto |
|----------|--------|
| `PLN_CHATBOT_KNOWLEDGE` | Ruta al JSON de conocimiento (por defecto `data/knowledge.default.json`). |
| `PLN_CHATBOT_SPACY_MODEL` | Modelo spaCy (por defecto `es_core_news_lg`). |
| `PLN_CHATBOT_LOG_LEVEL` | `INFO`, `DEBUG`, `WARNING`, etc. |
| `PLN_CHATBOT_TRACE` | `1` = mostrar trazas `DEBUG` en consola (por defecto). `0` / `false` = silenciar. |
| `PLN_CHATBOT_USE_RAG` | `1` = usar RAG si existe índice. `0` = desactivar. |
| `PLN_CHATBOT_RAG_INDEX` | Ruta al `.joblib` del índice. |
| `TELEGRAM_BOT_TOKEN` | Token de BotFather (solo para `telegram_bot`). **No subir a Git.** |

Un archivo `.env` en la raíz puede usarse en local si tu entorno lo carga; está en `.gitignore`.

---

## Estructura del repositorio

```text
Chatbot/
├── chatbot.py                 # Atajo → misma entrada que python -m pln_chatbot
├── morfosintactico.py         # (Opcional) Demo PLN independiente — ver sección «Material complementario»
├── requirements.txt           # Dependencias pip
├── README.md
├── data/
│   ├── knowledge.default.json # Base de conocimiento (ejemplo: computación / IA)
│   ├── knowledge.ventas.example.json
│   └── rag/
│       └── .gitkeep           # Aquí se genera rag_tech_index.joblib (local)
└── pln_chatbot/
    ├── __main__.py            # python -m pln_chatbot
    ├── cli.py                 # Bucle de consola
    ├── interaction.py         # Un turno: comandos + diálogo (consola / Telegram)
    ├── dialogue.py            # Intención, entidades y generación de respuesta
    ├── intents.py             # Gramática de rasgos NLTK (intenciones)
    ├── syntax_cfg.py          # CFG + ChartParser
    ├── extraction.py          # Extracción del tema en preguntas
    ├── morphology.py          # Salida de /morfologia
    ├── wordnet_tools.py       # Consultas WordNet
    ├── knowledge.py           # Carga y búsqueda del JSON
    ├── resources.py           # Inicialización NLTK, spaCy, CFG, RAG
    ├── config.py              # Rutas y flags
    ├── nlp_utils.py           # Normalización y acrónimos
    ├── rag.py                 # Índice TF-IDF y respuestas RAG
    ├── rag_ingest.py          # Construcción del índice desde Hugging Face
    ├── debug_trace.py         # Trazas DEBUG (/trace)
    └── telegram_bot.py        # Bot Telegram
```

### Rol de cada módulo (resumen)

| Módulo | Para qué está |
|--------|----------------|
| `cli.py` | Lee líneas en consola y delega en `interaction`. |
| `interaction.py` | Punto único por turno: ayuda, comandos PLN, diálogo. |
| `dialogue.py` | Orquesta intención → respuesta (JSON, WordNet, RAG, saludos). |
| `intents.py` | Clasificación rápida por gramática de rasgos. |
| `syntax_cfg.py` | Validación sintáctica con CFG (demostración / depuración). |
| `extraction.py` | Saca el «tema» de frases como «qué es …». |
| `morphology.py` | Tabla morfológica para `/morfologia`. |
| `wordnet_tools.py` | Definiciones y sinónimos desde WordNet. |
| `knowledge.py` | Lee y busca en el JSON de dominio. |
| `resources.py` | Carga recursos al arrancar (NLTK, spaCy, índice RAG). |
| `rag.py` / `rag_ingest.py` | Búsqueda y construcción del índice RAG. |
| `telegram_bot.py` | Adaptador Telegram con los mismos handlers lógicos. |

---

## Material complementario: `morfosintactico.py`

En la raíz del repositorio puede existir el script **`morfosintactico.py`**. **No forma parte del chatbot**: el paquete `pln_chatbot` **no lo importa** y no hace falta ejecutarlo para usar consola ni Telegram.

| | Chatbot (`pln_chatbot`) | `morfosintactico.py` |
|---|-------------------------|----------------------|
| **Propósito** | Agente conversacional (respuestas, JSON, WordNet, RAG opcional). | **Demostración** de análisis morfosintáctico y sintáctico sobre frases de ejemplo. |
| **Entrada** | Diálogo continuo o comandos `/morfologia`, `/syntax`, etc. | Lista fija de frases en el propio script (o puedes editarla). |
| **Salida** | Respuestas al usuario. | Tablas POS, árboles CFG con [NLTK](https://www.nltk.org/), visualización de **dependencias** con [spaCy displacy](https://spacy.io/usage/visualizers). |
| **Dependencias** | Las de `requirements.txt` + modelo spaCy. | Mismas bases (**NLTK** + **spaCy**); ideal en **Jupyter** o [Google Colab](https://colab.research.google.com/) (el script detecta notebook y adapta la salida). |

**Para qué sirve como complemento:** practica en un solo archivo lo que el chatbot reparte en módulos (`morphology.py`, `syntax_cfg.py`, `intents.py`): tokenización, etiquetas morfosintácticas, gramática de rasgos y árbol de dependencias, sin el flujo de diálogo ni Telegram.

**Cómo ejecutarlo** (venv activado, `es_core_news_lg` instalado):

```powershell
python morfosintactico.py
```

En consola, las dependencias spaCy pueden guardarse como `dependencias_spacy_ultimo.html` en la carpeta actual.

**Nota sobre Git:** en algunas copias del proyecto este archivo está en `.gitignore` (material opcional o local). Si no lo ves en GitHub pero sí en tu disco, es esperado: el chatbot funciona igual sin él.

---

## Personalizar el dominio

1. Copia `data/knowledge.default.json` a otro nombre (por ejemplo `data/mi_dominio.json`).
2. Edita `domain_label`, mensajes de bienvenida y el objeto `entries` (clave normalizada → texto de respuesta).
3. Arranca apuntando al archivo:

```powershell
$env:PLN_CHATBOT_KNOWLEDGE="data\mi_dominio.json"
python -m pln_chatbot
```

---

## Ampliar el proyecto

Suele bastar con:

1. Añadir entradas al JSON o ampliar reglas en `intents.py` / `syntax_cfg.py`.
2. Ajustar `dialogue.py` si necesitas nuevas intenciones o fuentes de respuesta.
3. Regenerar el índice RAG si cambias corpus o filtros en `rag_ingest.py`.

---

## Resolución rápida de problemas

| Síntoma | Qué revisar |
|---------|-------------|
| `No module named 'pln_chatbot'` | Ejecutar desde la raíz del repo; usar `python -m pln_chatbot`. |
| Error al cargar spaCy | `python -m spacy download es_core_news_lg` en el mismo venv. |
| WordNet vacío o error NLTK | Descargas manuales de `punkt`, `wordnet`, `omw-1.4` (ver arriba). |
| RAG no responde | ¿Existe `data/rag/rag_tech_index.joblib`? ¿`PLN_CHATBOT_USE_RAG=1`? |
| Telegram no arranca | Variable `TELEGRAM_BOT_TOKEN` definida en la misma sesión de terminal. |
| Mucho texto `DEBUG` | `PLN_CHATBOT_TRACE=0` o comando `/trace off` en consola. |
| Push a Git rechaza archivos grandes | No subas `.joblib`, `.venv`, `.env`; están en `.gitignore`. |

---

## Privacidad y datos

- En **Telegram**, los mensajes pasan por los servidores de Telegram según sus [términos](https://telegram.org/privacy).
- **RAG** puede descargar corpus públicos desde Internet al ejecutar `rag_ingest`; revisa las licencias de los datasets en Hugging Face si redistribuyes el índice.
- No incluyas tokens, `.env` ni bases con datos personales en commits públicos.

---

## Clonar y contribuir

```powershell
git clone https://github.com/Heeber24/Chatbot.git
cd Chatbot
```

Luego sigue la sección **Entorno virtual e instalación**. Para subir cambios propios: `git add .` → `git commit -m "..."` → `git push` (sin `.venv`, tokens ni `*.joblib`).

---

## Licencia y uso

Revisa si el repositorio incluye un archivo `LICENSE`. El código de ejemplo y los JSON son punto de partida: adapta dominio, reglas y corpus a tu caso de uso.
