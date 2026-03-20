# Dataset Migration Plan

## Objetivo
Swarm Forge deja TinyShakespeare como benchmark principal y migra a un pipeline de corpus web real.

## Fases
1. OpenWebText como corpus intermedio
2. FineWeb sample-100BT como corpus de escala posterior

## Principios
- Mantener nanoGPT vanilla como baseline
- Usar tokenizer BPE tipo GPT-2/tiktoken
- Mantener split reproducible train/val
- Generar artefactos binarios reproducibles
- Documentar tamaño, número de documentos, tokens y comandos exactos

## Motivación
El benchmark toy ya permitió validar:
- ejecución GPU
- campaigns
- search
- sensibilidad por presupuesto

Pero ya no es suficiente para extraer conclusiones serias de entrenamiento.