#!/bin/bash

# Guarda el tiempo de inicio
start_time=$(date +%s)

# Ejecuta el comando
python resnet50.py 1 ../resnet50_DYB.xmodel

# Guarda el tiempo de finalización
end_time=$(date +%s)

# Calcula la diferencia de tiempo
execution_time=$((end_time - start_time))

# Inicializa nImagenes
nImagenes=1000

# Calcula los FPS
fps=$(echo "scale=2; $nImagenes / $execution_time" | bc)

echo "El comando se ejecutó en $execution_time segundos."
echo "FPS: $fps"