# Laboratorio de minería de datos

Proyecto académico que usa regresión lineal múltiple para estimar tres resultados a partir de sus datos:

- Precio del dólar
- Nivel de glucosa
- Consumo de energía eléctrica

Incluye entrenamiento, gráficas, evaluación de modelos y una aplicación web para probar predicciones.

## Empezar

Requiere Python 3.10 o posterior. Desde esta carpeta, instala las dependencias y entrena los modelos:

```bash
pip install -r requirements.txt
python entrenar.py
```

Después, inicia la aplicación:

```bash
streamlit run app.py
```

En la página, selecciona el escenario, ingresa sus variables y pulsa **Calcular predicción**.

## Qué encontrarás

```text
app.py                 Interfaz web
entrenar.py            Entrena los tres modelos
data/                  Conjuntos de datos CSV
src/                   Configuración y lógica compartida
modelos/               Modelos entrenados (.joblib)
graficas/              Gráficas generadas
resultados/            Métricas, coeficientes e informe
```

Los modelos se guardan al ejecutar `entrenar.py`. Esa misma ejecución actualiza las gráficas y crea en `resultados/` el resumen de métricas y el informe con los valores obtenidos.

## Datos utilizados

| Escenario | Variables de entrada | Resultado estimado |
|---|---|---|
| Dólar | Día, inflación, tasa de interés | Precio del dólar |
| Glucosa | Edad, IMC, actividad física | Nivel de glucosa |
| Energía | Temperatura, hora, día de la semana | Consumo de energía |

La glucosa es un ejemplo académico; la predicción no es un diagnóstico médico.

## Procedencia

Implementación creada para el laboratorio CRISP-DM. Los archivos CSV se copiaron de los datos proporcionados en el espacio de trabajo.
