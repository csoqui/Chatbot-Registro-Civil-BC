# Guion para video demostrativo del proyecto

## 1. Presentación inicial

Hola, mi nombre es **[tu nombre]** y en este video presento mi Proyecto Integrador de Aprendizaje de la materia de **Procesamiento de Lenguaje Natural**.

El proyecto consiste en un agente conversacional desarrollado en **Python**, adaptado al dominio del **Registro Civil de Baja California**.

El objetivo del chatbot es recibir preguntas del usuario, ya sea por texto o por voz, procesarlas con técnicas de PLN y responder de forma contextual usando una base de conocimiento local y un índice RAG construido a partir de un manual de procedimientos.

---

## 2. Explicación del proyecto base

El proyecto inició a partir de un chatbot base proporcionado para la materia. Este chatbot ya tenía una estructura modular en Python y permitía trabajar con procesamiento de lenguaje natural.

A partir de ese código base realicé varias adecuaciones para cumplir con la rúbrica del proyecto integrador.

Durante esta parte puedo mostrar brevemente la estructura del proyecto:

```text
chatbot.py
pln_chatbot/
data/
requirements.txt
README.md
Resumen.md
```

---

## 3. Dominio elegido

El dominio que elegí para adaptar el chatbot fue **Registro Civil de Baja California**.

Este dominio permite responder preguntas sobre actos y trámites del estado civil de las personas, como nacimiento, matrimonio, defunción, corrección de actas, reconocimiento de hijos, CURP, copias certificadas y otros procedimientos.

Archivo a mostrar:

```text
data/knowledge.registro_civil_bc.json
```

Explicación:

En este archivo JSON se encuentra la base de conocimiento principal. Cada entrada relaciona un concepto del dominio con una respuesta contextual.

Por ejemplo, si el usuario pregunta por matrimonio civil, acta de nacimiento o corrección de acta, el chatbot busca primero en esta base local.

---

## 4. Entrada de texto

El primer requisito funcional es que el agente conversacional acepte entrada de texto.

Para esto se usa la versión de consola, ejecutando el archivo principal del proyecto.

Comandos para mostrar en terminal:

```bash
cd ~/Documents/Chatbot
source venv/bin/activate
export PLN_CHATBOT_KNOWLEDGE="data/knowledge.registro_civil_bc.json"
export PLN_CHATBOT_USE_RAG=1
export PLN_CHATBOT_RAG_INDEX="data/rag/rag_registro_civil_bc_index.joblib"
venv/bin/python chatbot.py
```

Preguntas de prueba:

```text
acta de nacimiento
```

```text
matrimonio civil
```

Explicación:

Aquí se observa que el chatbot recibe texto, detecta el tema principal y responde con información del dominio Registro Civil de Baja California.

---

## 5. Entrada por voz

El segundo requisito importante es aceptar entrada por voz.

Para cumplirlo se agregó un módulo llamado:

```text
pln_chatbot/voice_input.py
```

Este módulo utiliza las librerías **SpeechRecognition** y **PyAudio** para capturar audio desde el micrófono y convertirlo a texto.

Explicación:

Este módulo escucha una frase del usuario, usa reconocimiento de voz en español y devuelve el texto reconocido.

Después, ese texto se envía al mismo pipeline de PLN que utiliza la entrada escrita.

Prueba en consola:

```text
/voz
```

Cuando aparezca:

```text
Escuchando... hable ahora.
```

Decir en voz alta:

```text
acta de nacimiento
```

Explicación:

Como se observa, el sistema transcribe la voz a texto y después procesa la consulta normalmente.

---

## 6. Procesamiento de Lenguaje Natural

El chatbot incorpora varias técnicas de **Procesamiento de Lenguaje Natural**.

Primero, normaliza la entrada del usuario. Después identifica la intención, extrae el tema principal y busca una respuesta en la base de conocimiento.

Se utilizan reglas, patrones, extracción de tema, búsqueda en JSON, WordNet y análisis morfosintáctico.

Archivo a mostrar:

```text
pln_chatbot/dialogue.py
```

Explicación:

En este archivo está la lógica principal del chatbot. Aquí se procesa la entrada del usuario, se detecta la intención y se genera la respuesta.

---

## 7. Uso de NLTK y WordNet

Para cumplir con el uso de recursos de PLN, el proyecto utiliza **NLTK**.

NLTK se usa para trabajar con gramáticas, análisis sintáctico y recursos lingüísticos como **WordNet**.

Prueba:

```text
/wordnet registro
```

Explicación:

Aquí el chatbot consulta WordNet para obtener información lingüística relacionada con el término ingresado, como definiciones y sinónimos.

---

## 8. Uso de spaCy

También se utiliza **spaCy** con el modelo en español:

```text
es_core_news_lg
```

spaCy permite analizar la oración, identificar categorías gramaticales, lemas y extraer el tema principal de la pregunta.

Prueba:

```text
/morfologia el acta de nacimiento es un documento oficial
```

Explicación:

Aquí se muestra el análisis morfológico de la frase, incluyendo tokens, lemas y categorías gramaticales.

También se puede probar:

```text
/syntax que es acta de nacimiento
```

Explicación:

Este comando muestra una validación sintáctica usando una gramática formal.

---

## 9. Base de conocimiento contextual

El chatbot ofrece respuestas contextuales dentro de un dominio específico.

En este caso, el dominio es **Registro Civil de Baja California**, por lo que las respuestas se enfocan en actos registrales, trámites y procedimientos relacionados.

Preguntas de prueba:

```text
corrección de acta
```

```text
registro de nacimiento
```

Explicación:

Estas respuestas provienen de la base de conocimiento local en formato JSON.

---

## 10. RAG propio para Registro Civil BC

Además de la base JSON, se implementó un **RAG local** para ampliar las respuestas.

El RAG se construyó a partir de un **Manual de Procedimientos de la Dirección del Registro Civil**.

Archivos a mostrar:

```text
data/rag/MANUAL DE PROCEDIMIENTOS DE LA DRC CON INDICE.doc
data/rag/registro_civil_bc/manual_registro_civil_bc.txt
data/rag/rag_registro_civil_bc_index.joblib
```

Explicación:

Primero se convirtió el documento de Word a texto plano. Después se dividió en fragmentos y se generó un índice TF-IDF usando scikit-learn.

Este índice permite recuperar fragmentos relevantes del manual cuando la pregunta del usuario no coincide exactamente con la base JSON.

Script a mostrar:

```text
pln_chatbot/rag_ingest_registro_civil.py
```

Explicación:

Este script realiza la ingesta documental: lee documentos locales, limpia el texto, lo divide en fragmentos y genera el índice RAG.

Preguntas de prueba para RAG:

```text
qué dice el manual sobre defunciones
```

```text
qué procedimiento indica el manual para registro de nacimiento
```

Explicación:

En esta respuesta el chatbot consulta el índice RAG y recupera información del manual de procedimientos.

---

## 11. Telegram

Además de la consola, el chatbot también funciona en **Telegram**.

Para esto se utiliza la librería **python-telegram-bot** y un token generado con BotFather.

Archivo a mostrar:

```text
pln_chatbot/telegram_bot.py
```

Explicación:

Este módulo recibe mensajes desde Telegram y los envía al mismo pipeline de procesamiento que usa la consola.

Comandos para ejecutar Telegram:

```bash
export TELEGRAM_BOT_TOKEN="TOKEN_DEL_BOT"
export PLN_CHATBOT_KNOWLEDGE="data/knowledge.registro_civil_bc.json"
export PLN_CHATBOT_USE_RAG=1
export PLN_CHATBOT_RAG_INDEX="data/rag/rag_registro_civil_bc_index.joblib"
venv/bin/python -m pln_chatbot.telegram_bot
```

Pruebas en Telegram:

```text
/start
```

```text
matrimonio civil
```

```text
qué dice el manual sobre defunciones
```

Explicación:

Aquí se observa que el bot responde desde Telegram utilizando la misma base de conocimiento y el mismo RAG del Registro Civil.

Importante: **No mostrar el token real en el video**.

---

## 12. Explicación del flujo general

El flujo general del chatbot es el siguiente:

Primero, el usuario ingresa una consulta por texto o por voz. Si es voz, el sistema convierte el audio a texto.

Después, la entrada se limpia y se procesa con técnicas de PLN. Se identifica la intención, se extrae el tema principal y se busca una respuesta.

Primero se consulta la base local JSON. Si no hay coincidencia exacta, el sistema intenta recursos lingüísticos y búsqueda aproximada. Si aún no encuentra una respuesta, consulta el índice RAG local construido a partir del manual de procedimientos.

Finalmente, el chatbot entrega una respuesta contextual al usuario, ya sea en consola o Telegram.

---

## 13. Archivos modificados

Archivo a mostrar:

```text
Resumen.md
```

Explicación:

En este archivo documenté el resumen del proyecto y las modificaciones realizadas.

Principales archivos modificados o creados:

```text
data/knowledge.registro_civil_bc.json
pln_chatbot/voice_input.py
pln_chatbot/cli.py
pln_chatbot/interaction.py
pln_chatbot/telegram_bot.py
pln_chatbot/rag.py
pln_chatbot/dialogue.py
pln_chatbot/rag_ingest_registro_civil.py
requirements.txt
README.md
Resumen.md
```

---

## 14. Cumplimiento de la rúbrica

Con esta implementación se cumplen los puntos principales de la rúbrica:

- **Programa desarrollado en Python**
- **Agente conversacional funcional**
- **Entrada por texto**
- **Entrada por voz**
- **Procesamiento de lenguaje natural**
- **Uso de NLTK**
- **Uso de spaCy**
- **Uso de WordNet**
- **Respuestas contextuales en un dominio específico**
- **Base de conocimiento propia**
- **RAG local usando un manual de procedimientos**
- **Funcionamiento en consola**
- **Funcionamiento adicional en Telegram**

---

## 15. Cierre del video

En conclusión, el proyecto demuestra cómo un chatbot en Python puede integrar entrada de texto, entrada por voz y técnicas de Procesamiento de Lenguaje Natural para responder preguntas dentro de un dominio específico.

En este caso, el dominio fue Registro Civil de Baja California, usando una base de conocimiento local y un RAG construido a partir de un manual de procedimientos.

Con esto se cumple la rúbrica solicitada para el Proyecto Integrador de Aprendizaje.

---

## Orden recomendado para grabar

1. Presentación del proyecto.
2. Mostrar `Resumen.md`.
3. Mostrar estructura de archivos.
4. Mostrar JSON del dominio.
5. Ejecutar consola.
6. Probar texto.
7. Probar voz.
8. Mostrar NLTK / WordNet.
9. Mostrar spaCy.
10. Probar RAG.
11. Probar Telegram.
12. Cerrar con cumplimiento de rúbrica.

---

## Recordatorios antes de grabar

- Aumentar zoom en VS Code y terminal con `Cmd + +`.
- Limpiar terminal con `clear`.
- No mostrar el token real de Telegram.
- Hacer una prueba corta en OBS para verificar audio y pantalla.
- Tener los comandos listos antes de iniciar.
