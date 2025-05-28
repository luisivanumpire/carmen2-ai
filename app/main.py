from flask import Flask, request, jsonify
from flask_cors import CORS
from werkzeug.utils import secure_filename
import os
import numpy as np
import pickle
import json
import shutil

from app.utils.processing import (
    extraer_texto_pdf,
    dividir_en_chunks,
    generar_embeddings,
    consultar_resumen,
    consultar_etiquetas,
    modelo_embeddings,
    construir_indice_faiss,
    buscar_chunks_relevantes,
    consultar_modelo,
    USE_FAISS
)

# --- Configuración ---
DATA_DIR = "data/embeddings"
DOCS_DIR = "data/docs"
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(DOCS_DIR, exist_ok=True)

CHUNKS_FILE = os.path.join(DATA_DIR, "chunks.pkl")
METADATA_FILE = os.path.join(DATA_DIR, "metadata.pkl")
EMBEDDINGS_FILE = os.path.join(DATA_DIR, "embeddings.npy")
RESUMENES_FILE = os.path.join(DATA_DIR, "resumenes.json")

# --- Cargar datos persistentes ---
chunks = pickle.load(open(CHUNKS_FILE, "rb")) if os.path.exists(CHUNKS_FILE) else []
metadata = pickle.load(open(METADATA_FILE, "rb")) if os.path.exists(METADATA_FILE) else []
embeddings = np.load(EMBEDDINGS_FILE) if os.path.exists(EMBEDDINGS_FILE) else np.empty((0, 384))
resumenes = json.load(open(RESUMENES_FILE, "r", encoding="utf-8")) if os.path.exists(RESUMENES_FILE) else {}

# Reconstruir índice FAISS si aplica
if USE_FAISS and embeddings.size:
    from app.utils import processing
    processing.faiss_index = construir_indice_faiss(embeddings)

# --- App Flask ---
app = Flask(__name__)
CORS(app)

@app.route("/api/chat_prompt", methods=["POST"])
def chat_prompt():
    pregunta = request.json.get("pregunta", "")
    respuesta = consultar_modelo(pregunta)
    return jsonify({"respuesta": respuesta.strip()})

@app.route("/api/chat_docs", methods=["POST"])
def chat_docs():
    pregunta = request.json.get("pregunta", "")
    if not embeddings.size:
        return jsonify({"respuesta": "(No hay documentos cargados)"})
    relevantes = buscar_chunks_relevantes(embeddings, pregunta, chunks, modelo_embeddings)
    contexto = "\n".join(relevantes)
    prompt = f"Usa el siguiente contexto para responder:\n\n{contexto}\n\nPregunta: {pregunta}\n\nRespuesta:"
    respuesta = consultar_modelo(prompt)
    return jsonify({"respuesta": respuesta.strip()})

@app.route("/api/upload_docs", methods=["POST"])
def upload_docs():
    global embeddings
    if "file" not in request.files:
        return jsonify({"error": "No se recibió archivo PDF"}), 400

    file = request.files["file"]
    nombre_seguro = secure_filename(file.filename)
    ruta_pdf = os.path.join(DOCS_DIR, nombre_seguro)
    file.save(ruta_pdf)

    texto_paginas = extraer_texto_pdf(ruta_pdf)
    nuevos_chunks = []
    nueva_metadata = []
    for pagina, texto in texto_paginas:
        partes = dividir_en_chunks(texto)
        nuevos_chunks.extend(partes)
        nueva_metadata.extend([{"documento": nombre_seguro, "pagina": pagina}] * len(partes))

    nuevos_embeddings = generar_embeddings(nuevos_chunks)
    texto_total = "\n".join([t for _, t in texto_paginas])
    resumen = consultar_resumen(texto_total)
    etiquetas = consultar_etiquetas(texto_total)

    chunks.extend(nuevos_chunks)
    metadata.extend(nueva_metadata)
    embeddings = np.vstack([embeddings, nuevos_embeddings]) if embeddings.size else nuevos_embeddings

    resumenes[nombre_seguro] = {
        "resumen": resumen.strip(),
        "etiquetas": etiquetas["etiquetas"],
        "fuente_etiquetas": etiquetas["fuente"]
    }

    pickle.dump(chunks, open(CHUNKS_FILE, "wb"))
    pickle.dump(metadata, open(METADATA_FILE, "wb"))
    np.save(EMBEDDINGS_FILE, embeddings)
    with open(RESUMENES_FILE, "w", encoding="utf-8") as f:
        json.dump(resumenes, f, ensure_ascii=False, indent=2)

    if USE_FAISS:
        from app.utils import processing
        processing.faiss_index = construir_indice_faiss(embeddings)

    return jsonify({"mensaje": "Documento procesado correctamente"})

@app.route("/api/delete_docs", methods=["DELETE"])
def delete_docs():
    global chunks, metadata, embeddings, resumenes
    chunks = []
    metadata = []
    embeddings = np.empty((0, 384))
    resumenes = {}

    for archivo in [CHUNKS_FILE, METADATA_FILE, EMBEDDINGS_FILE, RESUMENES_FILE]:
        if os.path.exists(archivo): os.remove(archivo)
    for archivo in os.listdir(DOCS_DIR):
        os.remove(os.path.join(DOCS_DIR, archivo))

    return jsonify({"mensaje": "Archivos eliminados."})

@app.route("/api/docs_status", methods=["GET"])
def docs_status():
    hay_docs = bool(chunks and embeddings.size)
    return jsonify({"disponible": hay_docs})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5001, debug=True)

