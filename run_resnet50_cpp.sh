#!/bin/bash

# Guarda el tiempo de inicio
start_time=$(date +%s)

# Ejecuta el comando
./resnet50_pt resnet50_DYB.xmodel ../targetKria 1000

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