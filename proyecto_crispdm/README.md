# Laboratorio CRISP-DM: regresión lineal múltiple

Implementación del laboratorio descrito en el enunciado: modela el precio del dólar, el nivel de glucosa y el consumo de energía. Cada escenario tiene un modelo de regresión lineal múltiple, métricas de evaluación, gráficas y una predicción disponible desde Streamlit.

## Procedencia

Este directorio contiene una implementación independiente creada para el enunciado del laboratorio. Los tres archivos CSV se copiaron de los datos disponibles en el espacio de trabajo. El código de los scripts de la implementación anterior no se importa desde este proyecto.

## Preparar el entorno

Se recomienda Python 3.10 o posterior.

```bash
python -m venv .venv
```

En Windows:

```powershell
.venv\Scripts\Activate.ps1
```

En macOS o Linux:

```bash
source .venv/bin/activate
```

Instala las dependencias y entrena los tres modelos:

```bash
pip install -r requirements.txt
python entrenar.py
```

Inicia la interfaz después del entrenamiento:

```bash
streamlit run app.py
```

La aplicación permite seleccionar el escenario, introducir sus variables y calcular la predicción. El panel inferior muestra la ecuación, MSE, RMSE, R² e importancia estandarizada.

## Estructura

```text
proyecto_crispdm/
├── app.py                       Interfaz de predicción con Streamlit
├── entrenar.py                  Punto de entrada para entrenar los modelos
├── requirements.txt             Dependencias
├── data/                         Tres conjuntos de datos CSV
├── src/
│   ├── configuracion.py          Metadatos y variables de los escenarios
│   └── entrenamiento.py          Preparación, evaluación, gráficas y exportación
├── modelos/                      Modelos serializados en formato joblib
├── graficas/                     Relaciones, correlaciones y real frente a estimado
└── resultados/                   Métricas, coeficientes e informe de ejecución
```

## Método y resultados

El flujo limpia filas incompletas y duplicadas, separa los datos en 80 % para entrenamiento y 20 % para prueba con semilla 42, ajusta `LinearRegression` y calcula MSE, RMSE y R². Las métricas requeridas por el enunciado son MSE y R² para dólar y glucosa, y RMSE y R² para energía; el programa guarda las tres métricas en todos los casos para facilitar la comparación.

La importancia se compara mediante el coeficiente estandarizado `coeficiente × desviación estándar de X / desviación estándar de y`. Así se consideran las diferentes unidades de edad, temperatura, inflación y demás predictores.

Al ejecutar `python entrenar.py`, el programa crea los modelos `.joblib`, las gráficas PNG, `resultados/resumen_metricas.csv`, los coeficientes por escenario e `resultados/informe_resultados.md`. Este informe se genera con los valores medidos en esa ejecución; no se incluyen métricas asumidas por adelantado.

## Fases CRISP-DM representadas

1. **Comprensión del problema:** predecir tres objetivos cuantitativos en contextos distintos.
2. **Comprensión de datos:** cargar los CSV y verificar columnas disponibles.
3. **Preparación:** seleccionar columnas, quitar faltantes y duplicados, separar entrenamiento y prueba.
4. **Modelado:** ajustar regresión lineal múltiple e interpretar sus coeficientes.
5. **Evaluación:** informar MSE, RMSE, R² e impacto estandarizado; revisar gráficas.
6. **Despliegue:** guardar cada modelo y metadatos con joblib y servirlos en Streamlit.

## Lectura de las métricas

- **MSE:** promedio de errores al cuadrado; un valor menor representa menos error.
- **RMSE:** raíz del MSE y expresada en las unidades del objetivo.
- **R²:** proporción de variabilidad explicada por el modelo; puede ser menor que cero en datos de prueba.

Los coeficientes representan asociaciones lineales dentro de los datos usados, no causalidad. El uso de `Hora` como número supone una tendencia recta y no representa por sí solo el ciclo entre las horas 24 y 1. El modelo de glucosa es una práctica académica, no una herramienta clínica.
