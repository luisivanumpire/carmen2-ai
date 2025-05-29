import os
import fitz  # PyMuPDF
import requests
import numpy as np
import boto3
import json
import re
import faiss
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import nltk

nltk.download('stopwords')
from nltk.corpus import stopwords

# Configuración
OLLAMA_SERVER = "http://localhost:11434"
USE_BEDROCK = False  # Cambiar a True si usas Bedrock
USE_FAISS = False     # Cambiar a False para usar similitud del coseno
BEDROCK_REGION = "us-west-2"
BEDROCK_MODEL_ID = "meta.llama3-1-8b-instruct-v1:0"

# Modelo de embeddings y stopwords
modelo_embeddings = SentenceTransformer("all-MiniLM-L6-v2")
stop_words = stopwords.words('spanish')

faiss_index = None  # índice FAISS global


def extraer_texto_pdf(path):
    doc = fitz.open(path)
    texto_paginas = []
    for num_pagina in range(doc.page_count):
        texto = doc[num_pagina].get_text()
        texto_paginas.append((num_pagina + 1, texto))
    return texto_paginas


def dividir_en_chunks(texto, tamano=150):
    palabras = texto.split()
    return [" ".join(palabras[i:i + tamano]) for i in range(0, len(palabras), tamano)]


def generar_embeddings(lista_chunks):
    return modelo_embeddings.encode(lista_chunks)


def consultar_bedrock(prompt):
    bedrock = boto3.client("bedrock-runtime", region_name=BEDROCK_REGION)
    body = {
        "prompt": prompt,
        "max_gen_len": 512,
        "temperature": 0.5,
        "top_p": 0.9
    }
    response = bedrock.invoke_model(
        modelId=BEDROCK_MODEL_ID,
        contentType="application/json",
        accept="application/json",
        body=json.dumps(body)
    )
    result = json.loads(response["body"].read())
    return result.get("generation", "[Respuesta no encontrada]")


def consultar_ollama(prompt):
    r = requests.post(
        f"{OLLAMA_SERVER}/api/generate",
        json={"model": "llama3.2:1b", "prompt": prompt, "stream": False}
    )
    if r.status_code == 200:
        return r.json().get("response", "")
    return "[Error en Ollama]"


def consultar_modelo(prompt):
    return consultar_bedrock(prompt) if USE_BEDROCK else consultar_ollama(prompt)


def limpiar_intro_y_cierre(texto):
    texto = re.sub(r"^¡?Hola[!¡]?.{0,200}?(?=\*\*1\.|\n\n|\*\*)", "", texto, flags=re.IGNORECASE | re.DOTALL)
    texto = re.sub(r"(¿Qué necesitas\?.*?|¿Necesitas ayuda.*?)$", "", texto, flags=re.IGNORECASE | re.DOTALL)
    return texto.strip()


def consultar_resumen(texto):
    prompt = f"[Sistema] Genera una descripción clara, breve y formal del siguiente documento, como si fuera una ficha bibliográfica en una biblioteca. Incluye de forma resumida el propósito del documento, su contenido general y enfoque.\n\n[Contenido]\n{texto}\n\n[Asistente] "
    return limpiar_intro_y_cierre(consultar_modelo(prompt))


def extraer_keywords_tf_idf(texto, n=3):
    vectorizer = TfidfVectorizer(stop_words=stop_words, max_features=1000)
    tfidf_matrix = vectorizer.fit_transform([texto])
    scores = zip(vectorizer.get_feature_names_out(), tfidf_matrix.toarray()[0])
    top = sorted(scores, key=lambda x: x[1], reverse=True)[:n]
    return [word for word, _ in top]


def consultar_etiquetas(texto):
    prompt = f"[Sistema] Extrae exactamente 3 etiquetas clave que representen el contenido del siguiente documento. Usa una sola palabra por etiqueta. No incluyas frases completas. Devuelve solo la lista de palabras.\n\n[Contenido]\n{texto}\n\n[Asistente] "
    respuesta = consultar_modelo(prompt)
    etiquetas = [e.strip(" *•-•") for e in respuesta.split("\n") if e.strip()]
    etiquetas_validas = [e for e in etiquetas if len(e.split()) == 1]
    if len(etiquetas_validas) < 3:
        return {"fuente": "tf-idf", "etiquetas": extraer_keywords_tf_idf(texto)}
    return {"fuente": "bedrock" if USE_BEDROCK else "llama3.2:1b", "etiquetas": etiquetas_validas[:3]}


def construir_indice_faiss(embeddings):
    dim = embeddings.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(embeddings.astype(np.float32))
    return index


def buscar_similares_faiss(query_embedding, top_k=5):
    global faiss_index
    if faiss_index is None:
        raise RuntimeError("El índice FAISS no ha sido construido.")
    D, I = faiss_index.search(np.array([query_embedding]).astype(np.float32), top_k)
    return I[0]


def buscar_similares_coseno(embeddings, query_embedding, top_k=5):
    sims = cosine_similarity([query_embedding], embeddings)[0]
    top_indices = np.argsort(sims)[::-1][:top_k]
    return top_indices


def buscar_chunks_relevantes(embeddings, query_texto, chunks, modelo_embeddings, top_k=5):
    query_embedding = modelo_embeddings.encode([query_texto])[0]
    if USE_FAISS:
        return [chunks[i] for i in buscar_similares_faiss(query_embedding, top_k)]
    else:
        return [chunks[i] for i in buscar_similares_coseno(embeddings, query_embedding, top_k)]

