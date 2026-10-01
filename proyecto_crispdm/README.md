# Laboratorio de minería de datos

El proyecto entrena tres modelos de regresión lineal múltiple y permite probarlos desde una aplicación web.

| Modelo | Entradas | Predicción |
|---|---|---|
| Dólar | Día, inflación y tasa de interés | Precio del dólar |
| Glucosa | Edad, IMC y actividad física | Nivel de glucosa |
| Energía | Temperatura, hora y día de la semana | Consumo eléctrico |

## Ejecutar

Desde esta carpeta, con Python 3.10 o posterior:

```bash
pip install -r requirements.txt
python entrenar.py
streamlit run app.py
```

`entrenar.py` lee los CSV de `data/`, ajusta cada modelo con el 80 % de los datos y evalúa las predicciones con el 20 % restante. Luego imprime los coeficientes y las métricas, y guarda los modelos y resultados. Al terminar, `app.py` carga esos modelos; selecciona un escenario, ingresa sus variables y pulsa **Calcular predicción**.

## Archivos importantes

- `src/configuracion.py`: nombres de los conjuntos de datos, variables de entrada y objetivos.
- `src/entrenamiento.py`: limpieza, entrenamiento, evaluación, gráficas, exportación e informe.
- `src/interfaz.py`: componentes visuales, campos de entrada y flujo de predicción.
- `assets/interfaz.css`: colores, tipografía y distribución de la aplicación.
- `modelos/`: archivos `.joblib` que utiliza la aplicación. Se crean al ejecutar `entrenar.py`.
- `graficas/`: por cada modelo se guardan relaciones entre variables, matriz de correlación y gráfico de valores reales frente a estimados.
- `resultados/`: métricas comparativas, coeficientes por modelo e informe de la ejecución.

La evaluación incluye MSE, RMSE y R². La importancia de las variables se ordena mediante coeficientes estandarizados para poder compararlas aunque usen escalas distintas.

La predicción de glucosa es únicamente académica y no constituye un diagnóstico médico. Los CSV se copiaron de los datos proporcionados para el laboratorio.
