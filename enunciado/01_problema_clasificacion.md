# 3. Definición del problema de clasificación multiclase

Las categorías temáticas proceden de **EuroSciVoc**, el Vocabulario Europeo de Ciencia. Es una jerarquía de más de mil conceptos, con hasta seis niveles, que cuelgan de siete raíces que siguen los dos primeros niveles de la clasificación de Campos de Investigación y Desarrollo de la OCDE.

Un proyecto puede tener asociados varios términos EuroSciVoc, y cada término constituye un camino dentro de esa jerarquía.

Esto deja abiertas dos decisiones: qué nivel de la jerarquía predecir y qué hacer con los proyectos que tienen varios códigos. Para que todos los equipos resuelvan el mismo problema y los resultados sean comparables, el proyecto básico fija ambas decisiones:

- **Objetivo: el segundo nivel de la jerarquía.** Para cada término EuroSciVoc de cada proyecto, recorred la jerarquía hacia arriba hasta encontrar su ancestro de nivel 2. Los códigos que no alcancen el segundo nivel deben descartarse.
- **Una etiqueta por proyecto.** Para cada proyecto, utilizad la *primera* categoría EuroSciVoc disponible para la que pueda extraerse un objetivo válido de segundo nivel. La versión multietiqueta queda como una de las extensiones propuestas y no forma parte del proyecto básico.

Después de aplicar los dos criterios, deben descartarse los proyectos para los que no haya sido posible obtener una categoría válida de segundo nivel.

> **Punto de control:** se conserva aproximadamente el **90,31 %** de los proyectos proporcionados.

Además de la categoría de segundo nivel, los proyectos disponen de su categoría de **primer nivel**. Esta información, con solo **seis clases** (una de las siete raíces no la usa ningún proyecto), se utilizará como control; véase la Sección 4.4.

## Advertencia sobre las etiquetas

Los términos EuroSciVoc de cada proyecto fueron sugeridos mediante herramientas automáticas de procesamiento del lenguaje aplicadas a las descripciones de los proyectos y posteriormente validados por expertos humanos. Es decir, parte del *ground truth* que vais a predecir fue generado a partir del mismo texto que utilizaréis como entrada. Esto no invalida el ejercicio (las etiquetas han sido revisadas y son las que realmente utiliza CORDIS), pero una puntuación muy alta debería haceros sospechar que parte del éxito se debe a la forma en que se generaron las etiquetas.

## El coste de la simplificación

Convertir un problema multietiqueta en uno de etiqueta única es una decisión de modelado y afecta a la interpretación de los resultados.

- Solo alrededor del **32 %** de los proyectos tienen todos sus términos bajo la misma categoría de segundo nivel; los dos tercios restantes abarcan dos o más áreas, y la etiqueta es simplemente la primera.
- Si se usara «el ancestro de segundo nivel más frecuente» en lugar de «el primer término», la etiqueta cambiaría en aproximadamente el **12 %** de los proyectos.
- En el primer nivel: solo el **47 %** de los proyectos son inequívocos y, de nuevo, ~**12 %** de las etiquetas cambiarían con la regla de la mayoría.

Dos consecuencias para vuestra discusión:

1. **Existe un techo.** Una parte considerable de cualquier desacuerdo entre vuestro clasificador y la etiqueta se debe a la arbitrariedad de escoger una de las varias áreas a las que el proyecto pertenece realmente. No esperéis ni persigáis tasas cercanas al 100 %.
2. **Las confusiones son informativas.** Al analizar dónde fallan los modelos (Sección 4.4), podéis medir con qué frecuencia la clase predicha es una de las categorías de segundo nivel *descartadas*. Esta medida ayuda a distinguir cuánto del error se debe a la ambigüedad de la etiqueta y cuánto a un fallo real del modelo. Es uno de los valores más interesantes que podéis reportar.
