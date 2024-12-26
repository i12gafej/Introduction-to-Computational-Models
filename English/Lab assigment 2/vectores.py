import re
from collections import defaultdict

# Expresiones regulares
seed_pattern = re.compile(r"SEED (\d+)")
iteration_pattern = re.compile(r"Iteration (\d+)\s+Training error:\s+([\d.]+)\s+Test error:\s+([\d.]+)")

# Datos de entrada (sustituye esto por tu archivo o cadena completa) que leo del archivo resultado.txt
with open("resultado.txt") as f:
    data = f.read()

# Almacenar resultados
results = {}

# Variables auxiliares
current_seed = None

# Procesar línea por línea
for line in data.splitlines():
    # Capturar la semilla
    seed_match = seed_pattern.search(line)
    if seed_match:
        current_seed = int(seed_match.group(1))
        # Inicializar listas para la nueva semilla
        results[current_seed] = {
            "iterations": [],
            "training_errors": [],
            "test_errors": []
        }
        continue

    # Capturar iteraciones, errores de entrenamiento y prueba
    if current_seed is not None:
        iteration_match = iteration_pattern.search(line)
        if iteration_match:
            iteration = int(iteration_match.group(1))
            training_error = float(iteration_match.group(2))
            test_error = float(iteration_match.group(3))
            
            # Añadir los valores a las listas correspondientes
            results[current_seed]["iterations"].append(iteration)
            results[current_seed]["training_errors"].append(training_error)
            results[current_seed]["test_errors"].append(test_error)
