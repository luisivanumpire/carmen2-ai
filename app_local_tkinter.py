import tkinter as tk
from tkinter import filedialog, messagebox
import threading

from app.utils.processing import (
    extraer_texto_pdf,
    dividir_en_chunks,
    generar_embeddings,
    consultar_resumen,
    consultar_etiquetas,
    consultar_modelo,
    buscar_chunks_relevantes,
    modelo_embeddings,
    USE_FAISS
)

import numpy as np
import pickle
import os

# Config
DATA_DIR = "data/embeddings"
DOCS_DIR = "data/docs"
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(DOCS_DIR, exist_ok=True)

CHUNKS_FILE = os.path.join(DATA_DIR, "chunks.pkl")
EMBEDDINGS_FILE = os.path.join(DATA_DIR, "embeddings.npy")

chunks = pickle.load(open(CHUNKS_FILE, "rb")) if os.path.exists(CHUNKS_FILE) else []
embeddings = np.load(EMBEDDINGS_FILE) if os.path.exists(EMBEDDINGS_FILE) else np.empty((0, 384))

class App:
    def __init__(self, root):
        self.root = root
        root.title("AI Docs Local")

        tk.Button(root, text="📄 Cargar PDF", command=self.cargar_pdf).pack(fill=tk.X, padx=10, pady=5)
        self.text_resumen = tk.Text(root, height=5)
        self.text_resumen.pack(fill=tk.BOTH, padx=10, pady=5)

        tk.Label(root, text="❓ Preguntar al documento").pack()
        self.entry_pregunta = tk.Text(root, height=4)
        self.entry_pregunta.pack(fill=tk.BOTH, padx=10, pady=5)
        tk.Button(root, text="Responder", command=self.preguntar).pack(pady=5)

        self.text_respuesta = tk.Text(root, height=10)
        self.text_respuesta.pack(fill=tk.BOTH, padx=10, pady=5)

    def cargar_pdf(self):
        filepath = filedialog.askopenfilename(filetypes=[("PDF Files", "*.pdf")])
        if not filepath:
            return

        def tarea():
            texto_paginas = extraer_texto_pdf(filepath)
            nombre = os.path.basename(filepath)

            nuevos_chunks = []
            for _, texto in texto_paginas:
                nuevos_chunks.extend(dividir_en_chunks(texto))
            nuevos_embeddings = generar_embeddings(nuevos_chunks)

            resumen = consultar_resumen("\n".join([t for _, t in texto_paginas]))
            etiquetas_data = consultar_etiquetas("\n".join([t for _, t in texto_paginas]))

            self.text_resumen.delete("1.0", tk.END)
            self.text_resumen.insert(tk.END, f"Resumen:\n{resumen}\n\nEtiquetas: {', '.join(etiquetas_data['etiquetas'])}")

            global chunks, embeddings
            chunks.extend(nuevos_chunks)
            embeddings = np.vstack([embeddings, nuevos_embeddings]) if embeddings.size else nuevos_embeddings

            pickle.dump(chunks, open(CHUNKS_FILE, "wb"))
            np.save(EMBEDDINGS_FILE, embeddings)

            messagebox.showinfo("Listo", "Documento cargado correctamente")

        threading.Thread(target=tarea).start()

    def preguntar(self):
        pregunta = self.entry_pregunta.get("1.0", tk.END).strip()
        if not pregunta:
            messagebox.showwarning("Error", "Debes escribir una pregunta")
            return

        def tarea():
            relevantes = buscar_chunks_relevantes(embeddings, pregunta, chunks, modelo_embeddings)
            contexto = "\n".join(relevantes)
            prompt = f"Usa el siguiente contexto para responder:\n\n{contexto}\n\nPregunta: {pregunta}\n\nRespuesta:"
            respuesta = consultar_modelo(prompt)

            self.text_respuesta.delete("1.0", tk.END)
            self.text_respuesta.insert(tk.END, respuesta.strip())

        threading.Thread(target=tarea).start()

if __name__ == "__main__":
    root = tk.Tk()
    app = App(root)
    root.mainloop()

