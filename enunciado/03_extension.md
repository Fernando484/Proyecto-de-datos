# 5. Extensión (0,75 puntos)

La extensión es libre: ampliad el proyecto básico en la dirección que consideréis más apropiada. Ideas, aproximadamente en orden creciente de ambición:

- **¿Qué aporta el texto?** Los metadatos (coste, contribución de la UE, duración, país coordinador, número de países participantes) no tienen relación directa con el resumen. Construid un clasificador solo con metadatos, otro solo con texto y un tercero que combine ambos. ¿Cuánto de lo conseguido por el modelo de texto podría haberse obtenido sin los resúmenes?
- **Clasificación zero-shot con un LLM.** Dado el resumen y la lista de categorías, un LLM puede responder sin ejemplos etiquetados. Comparad con el proyecto básico usando prompts y modelos servidos localmente (Qwen, Mistral, Gemma).
- **Clasificación few-shot.** Añadid unos pocos ejemplos al prompt y plantead la pregunta inversa: ¿cuántos ejemplos etiquetados necesita vuestro pipeline supervisado antes de superar a un modelo al que no se le ha dado ninguno? Como punto de comparación más económico, el pipeline zero-shot basado en NLI de Hugging Face (`zero-shot-classification`) resuelve la misma tarea con un modelo mucho más pequeño.
- **Clasificación multietiqueta.** Recuperad las etiquetas descartadas en la Sección 3. Cambia la capa de salida, la función de pérdida y las métricas (subset accuracy, F1 micro y macro) y la decisión del umbral.
- **¿Ayuda la jerarquía?** Comparad un clasificador plano sobre el nivel 2 con uno jerárquico que primero prediga la raíz y luego la subcategoría condicionada a ella. Resolvedlo con vuestros datos.
- **¿La estructura no supervisada recupera la taxonomía?** Aplicad topic modeling o clustering sobre los embeddings contextuales, sin etiquetas, y comparad con EuroSciVoc mediante Adjusted Rand Index o Normalized Mutual Information. Es más interesante analizar dónde discrepan que dónde coinciden.
- **Predicción de producción científica.** Tarea de regresión sobre el número de publicaciones o patentes. Es más difícil de lo que parece y el tamaño y duración del proyecto explican buena parte de la varianza: la pregunta interesante es qué aporta el resumen además de esas variables.
- **Comparación de embeddings contextuales.** Comparad varios Transformers, incluidos modelos específicos de dominio (SciBERT y similares), y estudiad el efecto sobre el rendimiento final.

## Aspectos a abordar en la opción zero-shot con LLM

- **El modelo puede responder fuera de la taxonomía.** Con decenas de categorías en el prompt puede inventar, fusionar o parafrasear etiquetas. Decidid cómo mapear una respuesta a una categoría válida e informad de la fracción de respuestas corregidas o rescatadas: es un resultado, no un error de implementación.
- **La inferencia es lenta.** No se pueden etiquetar 35.000 resúmenes en un portátil. Tomad una submuestra del test, indicad su tamaño y evaluad los modelos supervisados **exactamente sobre ese mismo subconjunto**.
- **La granularidad afecta aún más.** Ejecutad zero-shot en los niveles 1 y 2 y comparad la degradación con la de los modelos supervisados.

Se pueden servir localmente modelos cuantizados de 7–8B mediante Ollama o Transformers, o solicitar al profesor acceso a un servicio Ollama del departamento TSC-UC3M vía HTTP.

## Notas

- La lista es solo una sugerencia: se puede elegir cualquier otro tema dentro del ámbito de la asignatura. Se valorará la originalidad. Ante dudas, consultad al profesor.
- La extensión vale 0,75 puntos: **evitad extensiones excesivamente ambiciosas** que comprometan la entrega en plazo.
