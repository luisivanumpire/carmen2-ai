#!/bin/bash

# Activar entorno Conda
source ~/miniconda3/etc/profile.d/conda.sh
conda activate ai-docs

# Verificar si backend ya está corriendo
if pgrep -f "python3 -m app.main" > /dev/null; then
    echo "✅ Backend Flask ya está corriendo."
else
    echo "🚀 Iniciando backend Flask..."
    python3 -m app.main &
fi

# Verificar si servidor HTTP ya está corriendo
if pgrep -f "http.server" > /dev/null; then
    echo "✅ Servidor HTTP ya está corriendo."
else
    echo "🌐 Iniciando servidor HTTP en puerto 8080..."
    python3 -m http.server 8080 --directory www &
fi

# Abrir navegador si no está abierto
if ! xdg-open http://192.168.1.213:8080 2>/dev/null; then
    echo "🔗 Abre manualmente http://localhost:8080"
fi

