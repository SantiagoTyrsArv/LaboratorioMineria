# Laboratorio 1: minería de datos

Implementación del laboratorio de regresión lineal múltiple descrito en el enunciado. El proyecto entrena tres modelos, interpreta sus coeficientes, evalúa sus predicciones, exporta los modelos y ofrece una aplicación Streamlit para probarlos.

## Escenarios

| Ejercicio | Datos de entrada | Variable objetivo | Evaluación solicitada |
|---|---|---|---|
| Dólar | `Dia`, `Inflacion`, `Tasa_interes` | `Precio_Dolar` | MSE y R² |
| Glucosa | `Edad`, `IMC`, `Actividad_Fisica` | `Nivel_Glucosa` | MSE y R²; comparar impacto de variables |
| Energía | `Temperatura`, `Hora`, `Dia_Semana` | `Consumo_Energia` | RMSE y R²; identificar la variable de mayor impacto |

Los datos de cada escenario están en `data/`. Para comparar variables que usan escalas distintas, el entrenamiento calcula coeficientes estandarizados.

## Instalación y uso

Se requiere Python 3.10 o posterior. Desde la carpeta `proyecto_crispdm`, ejecuta:

```bash
pip install -r requirements.txt
python entrenar.py
streamlit run app.py
```

Primero se entrenan los modelos; después se inicia la aplicación. En la interfaz, selecciona Dólar, Glucosa o Energía, ingresa los valores de sus variables y pulsa **Calcular estimación**. La página presenta el resultado y permite consultar las métricas, la ecuación y la importancia de cada variable.

## Proceso CRISP-DM

1. **Comprensión del problema:** estimar el precio del dólar, el nivel de glucosa y el consumo eléctrico.
2. **Comprensión de los datos:** carga los CSV y comprueba que estén las columnas necesarias.
3. **Preparación:** elimina filas con datos faltantes y duplicados; separa los datos en entrenamiento (80 %) y prueba (20 %).
4. **Modelado:** ajusta una regresión lineal múltiple para cada escenario e imprime el intercepto y los coeficientes.
5. **Evaluación:** calcula MSE, RMSE y R² sobre el conjunto de prueba. Los coeficientes estandarizados ordenan el impacto relativo de los predictores.
6. **Despliegue:** guarda los modelos y sus metadatos con joblib; la aplicación los carga para calcular predicciones.

## Archivos y resultados

- `entrenar.py`: ejecuta el entrenamiento de los tres escenarios.
- `src/configuracion.py`: define datasets, predictores, objetivos y unidades.
- `src/entrenamiento.py`: implementa preparación, ajuste, evaluación, gráficas, exportación e informe.
- `src/interfaz.py` y `assets/interfaz.css`: organizan los componentes y el diseño de Streamlit.
- `modelos/`: al entrenar, recibe `dolar.joblib`, `glucosa.joblib` y `energia.joblib`.
- `graficas/`: recibe gráficos de relaciones entre predictores y objetivo, correlaciones y valores reales frente a estimados.
- `resultados/`: recibe métricas, coeficientes por escenario y un informe generado con los resultados de la ejecución.

Los archivos de salida se actualizan al volver a ejecutar `python entrenar.py`. Las predicciones de glucosa son de carácter académico y no constituyen diagnósticos médicos.
