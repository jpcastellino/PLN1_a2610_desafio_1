# Desafío 1 - Procesamiento de Lenguaje Natural I

Vectorización de texto, similaridad coseno y clasificación Naïve Bayes sobre el dataset **20 Newsgroups** (scikit-learn).

## Contenido

- `a2610_Desafio_1.ipynb`: notebook con el desarrollo, salidas y análisis.

## Qué se hace

1. **Similaridad entre documentos:** 5 documentos al azar y sus 5 más similares (TF-IDF + coseno), analizando clases y limitaciones del método.
2. **Clasificador por prototipos (1-NN con coseno):** cada documento de test toma la clase del documento de train más similar.
3. **Naïve Bayes:** comparación de `MultinomialNB` y `ComplementNB` variando vectorizador y `alpha`, con selección de hiperparámetros sobre un conjunto de validación y evaluación final única en test (F1 macro).
4. **Similaridad entre palabras:** matriz término-documento transpuesta y 5 palabras elegidas manualmente.

Anexo: gráfico de la fórmula TF-IDF según la frecuencia de la palabra (lineal vs. `sublinear_tf`).

## Ejecución

```bash
pip install numpy pandas scikit-learn matplotlib jupyter
jupyter notebook a2610_Desafio_1.ipynb
```

La primera ejecución descarga el dataset 20 Newsgroups.
