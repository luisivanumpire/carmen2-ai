
# 🤖 Carmen1 AI – Plataforma Dual: Prompt & RAG con Documentos

Esta plataforma permite trabajar con una IA privada en dos modos:

- **Modo Prompt**: envía directamente tus preguntas al modelo LLaMA 3.2 1b vía Ollama.
- **Modo Docs-IA**: permite cargar documentos PDF y hacer preguntas que se responden con contexto (RAG).

---

## 🧠 Funcionalidades

### Modo 1: Prompt
- Envío directo del prompt al modelo de lenguaje.
- No requiere documentos.

### Modo 2: Docs-IA (RAG)
- Carga de documentos PDF.
- Generación automática de:
  - chunks
  - embeddings
  - resumen
  - etiquetas clave
- Preguntas usando recuperación semántica.
- Eliminación de todos los datos cargados.

---

## 🧰 Tecnologías usadas

- Python, Flask
- FAISS + SentenceTransformers
- Ollama + LLaMA 3.2:1b
- Frontend en HTML/JS
- Conda para entorno

---

## ⚙️ Instalación

```bash
git clone https://github.com/luisivanumpire/carmen1-ai.git
cd carmen1-ai
conda create -n ai-docs python=3.10 -y
conda activate ai-docs
pip install -r requirements.txt
chmod +x start.sh stop.sh
./start.sh
```

---

## 🌐 Interfaz

- Inicia en [http://localhost:8080](http://localhost:8080)
- Alterna entre los modos desde el menú superior.
- Usa “Adicionar archivos” para cargar PDFs.
- Usa “Eliminar archivos” para borrar todo.

---

## 📁 Estructura

```
carmen1-ai/
├── app/
│   ├── main.py              # Backend con API
│   └── utils/processing.py  # Funciones principales
├── data/                    # PDFs y embeddings
├── www/index.html           # Interfaz de usuario
├── start.sh / stop.sh       # Scripts de control
└── requirements.txt
```

---

## 📝 Licencia

MIT License
