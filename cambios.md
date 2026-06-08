# Cambios realizados al chatbot base

Este documento resume las adecuaciones realizadas al código original proporcionado por el maestro para cumplir con la rúbrica del Proyecto Integrador de Aprendizaje de Procesamiento de Lenguaje Natural.

## 1. Configuración inicial del proyecto

Se clonó y configuró el proyecto base en la carpeta `Documents/Chatbot`.

Se creó un entorno virtual de Python y se instalaron las dependencias del proyecto:

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

También se instaló el modelo de spaCy para español:

```bash
python -m spacy download es_core_news_lg
```

Con esto se validó que el chatbot original funcionara correctamente en consola.

## 2. Cambio de dominio del chatbot

El chatbot original estaba orientado a temas de Ciencias de la Computación e Inteligencia Artificial.

Se migró el dominio hacia un tema específico:

```text
Registro Civil de Baja California
```

Para esto se creó el archivo:

```text
data/knowledge.registro_civil_bc.json
```

Este archivo contiene la nueva base de conocimiento del chatbot, con temas como:

- Acta de nacimiento
- Copia certificada
- Matrimonio civil
- Requisitos para matrimonio
- Acta de defunción
- Corrección de acta
- CURP
- Citas
- Costos
- Oficialía
- Registro extemporáneo
- Reconocimiento de hijos

La base de conocimiento está en formato JSON, donde cada entrada relaciona un concepto con una respuesta contextual dentro del dominio del Registro Civil.

## 3. Carga del nuevo dominio mediante variable de entorno

No se eliminó la base original del proyecto. Se configuró el chatbot para cargar la nueva base mediante una variable de entorno:

```bash
export PLN_CHATBOT_KNOWLEDGE="data/knowledge.registro_civil_bc.json"
```

También se desactivó el RAG original:

```bash
export PLN_CHATBOT_USE_RAG=0
```

Esto se hizo porque el índice RAG original estaba enfocado en temas técnicos y podía generar respuestas fuera del dominio de Registro Civil.

## 4. Ajuste de bienvenida dinámica en Telegram

Se modificó el archivo:

```text
pln_chatbot/telegram_bot.py
```

Antes, el mensaje de bienvenida de Telegram estaba fijo con contenido relacionado con informática.

Después del cambio, la bienvenida se genera dinámicamente usando los datos de la base de conocimiento cargada:

- `assistant_name`
- `domain_label`
- `welcome_hint`

Esto permite que el bot de Telegram se adapte al dominio activo. Si se carga la base de Registro Civil BC, la bienvenida también corresponde a ese dominio.

## 5. Implementación de entrada por voz

Este fue uno de los cambios principales para cumplir la rúbrica.

Se creó el archivo:

```text
pln_chatbot/voice_input.py
```

Este módulo permite capturar audio desde el micrófono y convertirlo a texto usando las librerías:

- `SpeechRecognition`
- `PyAudio`

La función principal del módulo escucha una frase del usuario, intenta reconocerla en español de México y devuelve el texto reconocido.

Si ocurre un error, el módulo devuelve un mensaje entendible, por ejemplo:

- Falta instalar SpeechRecognition
- No se detectó micrófono
- No se entendió el audio
- Falló el servicio de reconocimiento de voz

## 6. Integración del comando `/voz` en consola

Se modificó el archivo:

```text
pln_chatbot/cli.py
```

Ahí se agregó el comando:

```text
/voz
```

Cuando el usuario escribe `/voz` en la consola:

1. El programa activa el micrófono.
2. Muestra el mensaje `Escuchando... hable ahora.`
3. Captura la voz del usuario.
4. Convierte el audio a texto.
5. Muestra en pantalla lo que entendió con `Usted dijo:`.
6. Envía ese texto al mismo pipeline de PLN que usa la entrada escrita.

De esta forma, el chatbot puede recibir entrada escrita y entrada por voz.

## 7. Actualización de comandos de ayuda

Se modificó el archivo:

```text
pln_chatbot/interaction.py
```

Se agregó el comando `/voz` al texto de ayuda del chatbot.

Ahora, cuando el usuario escribe:

```text
/ayuda
```

puede ver que `/voz` está disponible para capturar una pregunta mediante el micrófono.

## 8. Actualización de dependencias

Se modificó el archivo:

```text
requirements.txt
```

Se agregaron las siguientes dependencias:

```text
SpeechRecognition==3.10.4
PyAudio==0.2.14
```

Estas librerías permiten implementar la entrada por voz en la versión de consola.

## 9. Actualización de documentación

Se modificó el archivo:

```text
README.md
```

Se documentó que el chatbot ahora acepta:

- Entrada por texto en consola
- Entrada por voz en consola
- Entrada por texto en Telegram

También se agregó una nota para macOS, indicando que si `PyAudio` falla puede ser necesario instalar `portaudio`:

```bash
brew install portaudio
pip install PyAudio
```

Además, se documentó que macOS puede solicitar permisos de micrófono para Terminal, iTerm o el IDE desde donde se ejecute el programa.

## 10. Uso de recursos de PLN

El chatbot conserva y utiliza recursos de Procesamiento de Lenguaje Natural, como:

- NLTK
- spaCy
- WordNet
- Reglas y patrones
- Análisis morfosintáctico
- Búsqueda en base de conocimiento JSON

El texto reconocido por voz se procesa igual que el texto escrito, por lo que también pasa por estas técnicas de PLN.

## 11. Ejecución recomendada en consola

Para ejecutar el chatbot con el dominio de Registro Civil BC:

```bash
cd ~/Documents/Chatbot
source venv/bin/activate
export PLN_CHATBOT_KNOWLEDGE="data/knowledge.registro_civil_bc.json"
export PLN_CHATBOT_USE_RAG=0
python chatbot.py
```

Ejemplos de prueba por texto:

```text
acta de nacimiento
```

```text
requisitos para matrimonio
```

```text
corrección de acta
```

Ejemplo de prueba por voz:

```text
/voz
```

Después de escribir `/voz`, hablar al micrófono, por ejemplo:

```text
acta de nacimiento
```

El chatbot debe reconocer la frase y responder con información del Registro Civil.

## 12. Ejecución recomendada en Telegram

Para ejecutar el bot de Telegram con el dominio de Registro Civil BC:

```bash
cd ~/Documents/Chatbot
source venv/bin/activate
export TELEGRAM_BOT_TOKEN="TOKEN_REAL_DEL_BOT"
export PLN_CHATBOT_KNOWLEDGE="data/knowledge.registro_civil_bc.json"
export PLN_CHATBOT_USE_RAG=0
python -m pln_chatbot.telegram_bot
```

En Telegram se puede probar con:

```text
/start
```

```text
acta de nacimiento
```

```text
matrimonio civil
```

La entrada por voz se implementó principalmente para la consola, que es donde se cumple el requisito de la rúbrica.

## 13. Guion breve para el video demostrativo

Para el video se puede explicar lo siguiente:

```text
El proyecto base fue proporcionado por el maestro para trabajar un chatbot con Procesamiento de Lenguaje Natural. A partir de ese código realicé varias adecuaciones para cumplir con la rúbrica del proyecto integrador.

Primero configuré el entorno virtual, instalé las dependencias y descargué el modelo de spaCy en español. Después verifiqué que el chatbot original funcionara correctamente en consola.

Posteriormente migré el dominio del chatbot. El proyecto original respondía sobre temas de informática e inteligencia artificial, y lo adapté al dominio de Registro Civil de Baja California. Para eso creé una nueva base de conocimiento en formato JSON con temas como actas de nacimiento, matrimonio civil, CURP, citas, costos y corrección de actas.

También configuré el programa para cargar esta nueva base mediante variables de entorno y desactivé el RAG original para evitar respuestas de temas técnicos.

Después adapté el módulo de Telegram para que la bienvenida no fuera fija, sino dinámica, tomando el nombre del asistente y el dominio desde la base de conocimiento activa.

El cambio más importante fue la implementación de entrada por voz. Para esto agregué un nuevo módulo llamado voice_input.py, que utiliza SpeechRecognition y PyAudio para capturar audio desde el micrófono y convertirlo a texto. Luego integré el comando /voz en la consola, de manera que el usuario puede hablar y el texto reconocido se procesa con el mismo pipeline de PLN que una entrada escrita.

Finalmente actualicé las dependencias, la ayuda del chatbot y el README para documentar el uso de voz, los comandos y las instrucciones de ejecución.
```

## 14. Puntos de la rúbrica que se cumplen

- Programa desarrollado en Python.
- Agente conversacional funcional.
- Entrada por texto.
- Entrada por voz mediante micrófono.
- Procesamiento de lenguaje natural.
- Uso de NLTK.
- Uso de spaCy.
- Uso de WordNet.
- Respuestas contextuales dentro de un dominio específico.
- Dominio adaptado a Registro Civil de Baja California.
- Código organizado por módulos.
- Documentación actualizada.
- Funcionamiento en consola.
- Funcionamiento adicional en Telegram.
