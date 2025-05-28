#!/bin/bash

echo "Deteniendo backend Flask..."
pkill -f "python3 -m app.main"

echo "Deteniendo servidor HTTP..."
pkill -f "http.server"

echo "Todos los servicios han sido detenidos."

