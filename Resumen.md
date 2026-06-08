# Resumen del proyecto realizado

Para obtener este proyecto partimos de un **chatbot base en Python** proporcionado para la materia y lo adaptamos para cumplir con la rúbrica del Proyecto Integrador de Aprendizaje de Procesamiento de Lenguaje Natural.

El resultado final es un **chatbot conversacional especializado en el dominio del Registro Civil de Baja California**, capaz de funcionar en consola, aceptar entrada por voz, responder en Telegram y consultar información adicional desde un manual de procedimientos mediante RAG.

## 1. Preparación del proyecto

Primero se trabajó sobre el proyecto base clonado en:

```text
/Users/csoqui/Documents/Chatbot
```

Se creó y usó un entorno virtual de Python:

```text
venv
```

La versión utilizada fue:

```text
Python 3.9.6
```

También se instalaron las dependencias necesarias para PLN, Telegram, RAG y voz.

## 2. Cambio de dominio del chatbot

El chatbot original estaba orientado a temas de:

```text
Ciencias de la computación e IA
```

Nosotros lo adaptamos al dominio:

```text
Registro Civil de Baja California
```

Para esto se creó una nueva base de conocimiento:

```text
data/knowledge.registro_civil_bc.json
```

En ese archivo se agregaron conceptos como:

- **Registro Civil**
- **Acta de nacimiento**
- **Copia certificada**
- **Matrimonio civil**
- **Acta de defunción**
- **Corrección de acta**
- **Aclaración de acta**
- **Registro extemporáneo**
- **Reconocimiento de hijos**
- **CURP**
- **Apostilla**
- **Legalización**

Esta base permite que el chatbot responda preguntas específicas del dominio elegido.

## 3. Configuración dinámica del dominio

Se aprovechó la variable de entorno:

```bash
PLN_CHATBOT_KNOWLEDGE
```

para indicar qué base de conocimiento debe cargar el chatbot.

Para usar el dominio de Registro Civil BC se ejecuta:

```bash
export PLN_CHATBOT_KNOWLEDGE="data/knowledge.registro_civil_bc.json"
```

Esto permite cambiar de dominio sin modificar directamente el código fuente.

## 4. Entrada por texto en consola

Se mantuvo la entrada tradicional por texto en consola.

El chatbot recibe la consulta del usuario, por ejemplo:

```text
acta de nacimiento
```

Luego procesa la oración con técnicas de PLN y busca una respuesta contextual en la base de conocimiento local.

El flujo básico es:

```text
Entrada del usuario
→ procesamiento lingüístico
→ extracción del tema principal
→ búsqueda en la base local
→ generación de respuesta
```

## 5. Entrada por voz en consola

Se agregó soporte para entrada por voz en la consola.

Para eso se creó el módulo:

```text
pln_chatbot/voice_input.py
```

Este módulo usa:

```text
SpeechRecognition
PyAudio
```

La funcionalidad permite escribir en consola:

```text
/voz
```

Después el sistema escucha por el micrófono, transcribe la voz a texto y envía ese texto al mismo pipeline de procesamiento del chatbot.

También se modificó:

```text
pln_chatbot/cli.py
```

para integrar el comando `/voz`.

Además se actualizó:

```text
pln_chatbot/interaction.py
```

para que `/ayuda` muestre el nuevo comando de voz.

## 6. Dependencias agregadas

Se actualizaron las dependencias del proyecto en:

```text
requirements.txt
```

Se agregaron:

```text
SpeechRecognition==3.10.4
PyAudio==0.2.14
```

Estas librerías permiten capturar audio desde el micrófono y convertirlo en texto.

En macOS también se consideró la instalación de:

```bash
brew install portaudio
```

en caso de problemas con `PyAudio`.

## 7. Procesamiento de Lenguaje Natural

El chatbot utiliza varias técnicas de PLN.

### NLTK

Se usa para:

- **Tokenización**
- **Gramáticas**
- **Análisis sintáctico**
- **WordNet**

Ejemplo de comando:

```text
/wordnet registro
```

### spaCy

Se usa con el modelo:

```text
es_core_news_lg
```

para:

- **Lematización**
- **Análisis morfológico**
- **Categorías gramaticales**
- **Extracción del tema principal**

Ejemplo:

```text
/morfologia el acta de nacimiento es un documento oficial
```

### CFG

También se usa una gramática formal para validar ciertas estructuras de preguntas.

Por ejemplo, cuando aparece:

```text
DEBUG SINTAXIS_CFG
```

significa que el sistema intentó analizar la estructura gramatical de la oración.

## 8. Integración con Telegram

El chatbot también se configuró para funcionar como bot de Telegram.

Se trabajó con el archivo:

```text
pln_chatbot/telegram_bot.py
```

Se ajustó el mensaje de bienvenida para que sea dinámico y use la información del dominio cargado.

Antes el mensaje estaba orientado a informática, pero ahora puede mostrar:

```text
Asistente Registro Civil BC
Registro Civil de Baja California
```

Para ejecutar Telegram se usa:

```bash
export TELEGRAM_BOT_TOKEN="TOKEN_DEL_BOT"
export PLN_CHATBOT_KNOWLEDGE="data/knowledge.registro_civil_bc.json"
export PLN_CHATBOT_USE_RAG=1
export PLN_CHATBOT_RAG_INDEX="data/rag/rag_registro_civil_bc_index.joblib"

python -m pln_chatbot.telegram_bot
```

## 9. Creación del RAG para Registro Civil BC

También se creó un RAG local para ampliar las respuestas del chatbot usando un documento real.

El documento utilizado fue:

```text
MANUAL DE PROCEDIMIENTOS DE LA DRC CON INDICE.doc
```

Este documento corresponde al manual de procedimientos de la Dirección del Registro Civil.

## 10. Conversión del documento

El manual original estaba en formato `.doc`.

Se convirtió a texto plano para poder procesarlo:

```text
data/rag/registro_civil_bc/manual_registro_civil_bc.txt
```

Esto permitió que el contenido del manual pudiera ser leído, limpiado y fragmentado por el sistema.

## 11. Ingesta documental

Después se realizó un proceso de:

```text
Ingesta documental para RAG
```

Este proceso consistió en:

1. **Leer el documento convertido a texto**
2. **Limpiar el contenido**
3. **Dividirlo en fragmentos**
4. **Crear un índice de búsqueda**
5. **Guardar el índice para usarlo en el chatbot**

Se creó el script:

```text
pln_chatbot/rag_ingest_registro_civil.py
```

Ese script generó el índice:

```text
data/rag/rag_registro_civil_bc_index.joblib
```

El índice se generó correctamente con:

```text
224 fragmentos
```

## 12. Activación del RAG

Se modificó la lógica del chatbot para que pudiera usar el RAG del Registro Civil cuando el dominio activo fuera Registro Civil BC.

Se modificaron principalmente:

```text
pln_chatbot/rag.py
pln_chatbot/dialogue.py
```

En `rag.py` se agregaron palabras clave del dominio, como:

- **acta**
- **nacimiento**
- **matrimonio**
- **defunción**
- **oficialía**
- **corrección**
- **aclaración**
- **registro civil**
- **manual**
- **procedimiento**

Y en `dialogue.py` se ajustó la activación del RAG para que también responda consultas del Registro Civil.

## 13. Configuración del RAG

Para usar el RAG del manual se usan estas variables:

```bash
export PLN_CHATBOT_USE_RAG=1
export PLN_CHATBOT_RAG_INDEX="data/rag/rag_registro_civil_bc_index.joblib"
```

Con eso el chatbot carga el índice generado desde el manual.

## 14. Pruebas realizadas

Se probaron consultas desde consola como:

```text
acta de nacimiento
```

```text
matrimonio civil
```

```text
corrección de acta
```

```text
registro de nacimiento
```

También se probaron consultas orientadas al manual:

```text
qué dice el manual sobre defunciones
```

```text
qué procedimiento indica el manual para registro de nacimiento
```

```text
qué requisitos menciona el manual para registrar un nacimiento
```

## 15. Pruebas con voz

En consola se probó:

```text
/voz
```

Y se dictaron frases como:

```text
acta de nacimiento
```

```text
matrimonio civil
```

El sistema reconoció la voz, la convirtió a texto y procesó la consulta normalmente.

## 16. Pruebas en Telegram

También se probó el bot desde Telegram con comandos como:

```text
/start
```

y preguntas como:

```text
matrimonio civil
```

```text
qué dice el manual sobre defunciones
```

Esto permitió comprobar que Telegram usa el mismo procesamiento del chatbot.

## 17. Documentación

Se actualizó el archivo:

```text
README.md
```

para documentar:

- **Uso del chatbot**
- **Entrada por texto**
- **Entrada por voz**
- **Dependencias**
- **Notas para macOS**
- **Uso de Telegram**
- **Uso de RAG**

También se creó:

```text
cambios.md
```

donde se resumieron los cambios realizados al proyecto base.

## 18. Resultado final

El resultado final es un chatbot en Python que cumple con los elementos principales del proyecto:

- **Desarrollado en Python**
- **Funciona en consola**
- **Acepta entrada por texto**
- **Acepta entrada por voz en consola**
- **Procesa lenguaje natural**
- **Usa NLTK**
- **Usa spaCy**
- **Usa WordNet**
- **Tiene una base de conocimiento propia**
- **Está especializado en Registro Civil BC**
- **Tiene RAG local basado en un manual de procedimientos**
- **Funciona también en Telegram**

## Frase final para el video

```text
En conclusión, el proyecto consistió en adaptar un chatbot base en Python para convertirlo en un asistente especializado en el Registro Civil de Baja California. Se integró entrada por texto, entrada por voz en consola, procesamiento de lenguaje natural con NLTK, spaCy y WordNet, una base de conocimiento local en JSON, funcionamiento en Telegram y un sistema RAG construido a partir de un manual de procedimientos. Con esto el chatbot puede responder preguntas contextualizadas dentro del dominio seleccionado.
```
