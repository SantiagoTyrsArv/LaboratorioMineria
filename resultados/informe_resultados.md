# Informe de resultados - Laboratorio de regresión

## Método

Se siguieron las fases CRISP-DM: comprensión del problema, revisión de datos, limpieza de filas incompletas y duplicadas, modelado con regresión lineal múltiple, evaluación en una partición de prueba y exportación de modelos para despliegue.
La partición es 80/20 y usa `random_state=42` para permitir reproducirla.
La importancia relativa se estima con coeficientes estandarizados, calculados a partir de la desviación estándar de cada variable y del objetivo.

## Resultados

| Ejercicio | Filas válidas | MSE | RMSE | R² | Variable de mayor impacto |
|---|---:|---:|---:|---:|---|
| Precio del dólar | 500 | 2376.9709 | 48.7542 | 0.9963 | Dia |
| Nivel de glucosa | 2000 | 233.6930 | 15.2870 | 0.6814 | Edad |
| Consumo de energía | 10000 | 429.5187 | 20.7248 | 0.8968 | Temperatura |

## Precio del dólar

Objetivo: estimar `Precio_Dolar` en pesos.

Ecuación: `Precio_Dolar = 3985.7833 +4.9843 × Dia -870.7317 × Inflacion -1.3774 × Tasa_interes`

MSE = 2376.9709; RMSE = 48.7542; R² = 0.9963.

Los coeficientes describen el cambio esperado en el objetivo al aumentar una unidad la variable correspondiente, manteniendo las demás constantes. La comparación de impacto se basa en magnitudes estandarizadas.

La variable con mayor impacto estandarizado en esta ejecución es **Dia**.

Gráficas: `graficas/dolar_relaciones.png`, `graficas/dolar_correlaciones.png` y `graficas/dolar_real_estimado.png`.

## Nivel de glucosa

Objetivo: estimar `Nivel_Glucosa` en mg/dL.

Ecuación: `Nivel_Glucosa = 65.8609 +1.2266 × Edad +0.9334 × IMC -2.0853 × Actividad_Fisica`

MSE = 233.6930; RMSE = 15.2870; R² = 0.6814.

Los coeficientes describen el cambio esperado en el objetivo al aumentar una unidad la variable correspondiente, manteniendo las demás constantes. La comparación de impacto se basa en magnitudes estandarizadas.

La variable con mayor impacto estandarizado en esta ejecución es **Edad**.

Gráficas: `graficas/glucosa_relaciones.png`, `graficas/glucosa_correlaciones.png` y `graficas/glucosa_real_estimado.png`.

## Consumo de energía

Objetivo: estimar `Consumo_Energia` en kWh.

Ecuación: `Consumo_Energia = 101.2882 +9.9529 × Temperatura +5.0198 × Hora -3.0312 × Dia_Semana`

MSE = 429.5187; RMSE = 20.7248; R² = 0.8968.

Los coeficientes describen el cambio esperado en el objetivo al aumentar una unidad la variable correspondiente, manteniendo las demás constantes. La comparación de impacto se basa en magnitudes estandarizadas.

La variable con mayor impacto estandarizado en esta ejecución es **Temperatura**.

Gráficas: `graficas/energia_relaciones.png`, `graficas/energia_correlaciones.png` y `graficas/energia_real_estimado.png`.

## Limitaciones

Los resultados describen estos conjuntos de datos y no garantizan el mismo desempeño con datos nuevos. En particular, hora y día de la semana son variables cíclicas que una regresión lineal directa representa de forma aproximada. La predicción de glucosa es un ejercicio académico y no tiene uso diagnóstico.
