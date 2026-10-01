"""Preparación, ajuste, evaluación y publicación de modelos."""

from __future__ import annotations

from typing import Any

import joblib
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

from src.configuracion import (
    CARPETA_DATOS,
    CARPETA_GRAFICAS,
    CARPETA_MODELOS,
    CARPETA_RESULTADOS,
    EJERCICIOS,
    Ejercicio,
)


def cargar_tabla(ejercicio: Ejercicio) -> pd.DataFrame:
    """Lee un CSV y comprueba que contiene todas las columnas requeridas."""
    ruta = CARPETA_DATOS / ejercicio.archivo_csv
    if not ruta.is_file():
        raise FileNotFoundError(f"No se encontró el conjunto de datos: {ruta}")

    tabla = pd.read_csv(ruta)
    columnas = set(ejercicio.variables) | {ejercicio.objetivo}
    faltantes = columnas - set(tabla.columns)
    if faltantes:
        raise ValueError(f"{ruta.name} no contiene las columnas: {', '.join(sorted(faltantes))}")
    return tabla


def preparar_tabla(
    tabla: pd.DataFrame, ejercicio: Ejercicio
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]:
    """Limpia filas incompletas y crea particiones reproducibles 80/20."""
    columnas = [*ejercicio.variables, ejercicio.objetivo]
    limpia = tabla[columnas].dropna().drop_duplicates().copy()
    if len(limpia) < 5:
        raise ValueError(f"Se necesitan al menos cinco filas válidas para {ejercicio.nombre}.")

    entradas = limpia.loc[:, list(ejercicio.variables)]
    salida = limpia[ejercicio.objetivo]
    X_train, X_test, y_train, y_test = train_test_split(
        entradas, salida, test_size=0.2, random_state=42
    )
    return limpia, X_train, y_train, X_test, y_test


def calcular_importancia(
    modelo: LinearRegression,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    variables: tuple[str, ...],
) -> pd.DataFrame:
    """Ordena predictores por magnitud del coeficiente estandarizado."""
    desviacion_y = float(y_train.std())
    if desviacion_y == 0:
        betas = np.zeros(len(variables))
    else:
        betas = modelo.coef_ * X_train.std().to_numpy() / desviacion_y

    tabla = pd.DataFrame(
        {"Variable": variables, "Coeficiente": modelo.coef_, "Beta_estandarizado": betas}
    )
    tabla["Impacto_absoluto"] = tabla["Beta_estandarizado"].abs()
    return tabla.sort_values("Impacto_absoluto", ascending=False).reset_index(drop=True)


def crear_graficas(
    tabla: pd.DataFrame,
    ejercicio: Ejercicio,
    y_test: pd.Series,
    y_pred: np.ndarray,
) -> None:
    """Guarda dispersión por predictor, correlación y valores real/predicho."""
    CARPETA_GRAFICAS.mkdir(exist_ok=True)
    figura, ejes = plt.subplots(1, len(ejercicio.variables), figsize=(6 * len(ejercicio.variables), 4.8))
    for eje, variable in zip(np.atleast_1d(ejes), ejercicio.variables):
        sns.regplot(
            data=tabla,
            x=variable,
            y=ejercicio.objetivo,
            ax=eje,
            scatter_kws={"alpha": 0.35, "s": 14},
            line_kws={"color": "darkorange"},
        )
        eje.set_title(f"{variable} frente a {ejercicio.objetivo}")
    figura.tight_layout()
    figura.savefig(CARPETA_GRAFICAS / f"{ejercicio.clave}_relaciones.png", dpi=140)
    plt.close(figura)

    figura, eje = plt.subplots(figsize=(7, 5.5))
    sns.heatmap(tabla.corr(numeric_only=True), annot=True, fmt=".2f", cmap="vlag", center=0, ax=eje)
    eje.set_title(f"Correlaciones: {ejercicio.nombre}")
    figura.tight_layout()
    figura.savefig(CARPETA_GRAFICAS / f"{ejercicio.clave}_correlaciones.png", dpi=140)
    plt.close(figura)

    figura, eje = plt.subplots(figsize=(6, 6))
    eje.scatter(y_test, y_pred, alpha=0.45, s=18)
    limites = [min(float(y_test.min()), float(np.min(y_pred))), max(float(y_test.max()), float(np.max(y_pred)))]
    eje.plot(limites, limites, "k--", label="Estimación perfecta")
    eje.set(xlabel="Valor observado", ylabel="Valor estimado", title=f"Real y estimado: {ejercicio.nombre}")
    eje.legend()
    figura.tight_layout()
    figura.savefig(CARPETA_GRAFICAS / f"{ejercicio.clave}_real_estimado.png", dpi=140)
    plt.close(figura)


def entrenar_ejercicio(ejercicio: Ejercicio) -> dict[str, Any]:
    """Ajusta, mide, grafica y exporta el modelo de un escenario."""
    tabla = cargar_tabla(ejercicio)
    limpia, X_train, y_train, X_test, y_test = preparar_tabla(tabla, ejercicio)

    modelo = LinearRegression().fit(X_train, y_train)
    predicciones = modelo.predict(X_test)
    mse = float(mean_squared_error(y_test, predicciones))
    metricas = {
        "MSE": mse,
        "RMSE": float(np.sqrt(mse)),
        "R2": float(r2_score(y_test, predicciones)),
    }
    importancia = calcular_importancia(modelo, X_train, y_train, ejercicio.variables)
    crear_graficas(limpia, ejercicio, y_test, predicciones)

    CARPETA_MODELOS.mkdir(exist_ok=True)
    CARPETA_RESULTADOS.mkdir(exist_ok=True)
    paquete = {
        "modelo": modelo,
        "clave": ejercicio.clave,
        "variables": list(ejercicio.variables),
        "objetivo": ejercicio.objetivo,
        "unidad": ejercicio.unidad,
        "metricas": metricas,
        "importancia": importancia.to_dict(orient="records"),
    }
    joblib.dump(paquete, CARPETA_MODELOS / f"{ejercicio.clave}.joblib")
    importancia.to_csv(CARPETA_RESULTADOS / f"{ejercicio.clave}_coeficientes.csv", index=False)

    return {
        "ejercicio": ejercicio,
        "modelo": modelo,
        "metricas": metricas,
        "importancia": importancia,
        "filas_validas": len(limpia),
        "filas_prueba": len(y_test),
    }


def entrenar_todos() -> list[dict[str, Any]]:
    """Ejecuta los tres escenarios y guarda el informe tabular de métricas."""
    resultados = [entrenar_ejercicio(ejercicio) for ejercicio in EJERCICIOS]
    resumen = pd.DataFrame(
        [
            {
                "Ejercicio": resultado["ejercicio"].nombre,
                **resultado["metricas"],
                "Variable_mayor_impacto": resultado["importancia"].iloc[0]["Variable"],
            }
            for resultado in resultados
        ]
    )
    resumen.to_csv(CARPETA_RESULTADOS / "resumen_metricas.csv", index=False)
    crear_informe(resultados)
    return resultados


def crear_informe(resultados: list[dict[str, Any]]) -> None:
    """Genera un informe de resultados con las métricas de la ejecución actual."""
    lineas = [
        "# Informe de resultados - Laboratorio de regresión",
        "",
        "## Método",
        "",
        "Se siguieron las fases CRISP-DM: comprensión del problema, revisión de datos, "
        "limpieza de filas incompletas y duplicadas, modelado con regresión lineal múltiple, "
        "evaluación en una partición de prueba y exportación de modelos para despliegue.",
        "La partición es 80/20 y usa `random_state=42` para permitir reproducirla.",
        "La importancia relativa se estima con coeficientes estandarizados, calculados "
        "a partir de la desviación estándar de cada variable y del objetivo.",
        "",
        "## Resultados",
        "",
        "| Ejercicio | Filas válidas | MSE | RMSE | R² | Variable de mayor impacto |",
        "|---|---:|---:|---:|---:|---|",
    ]
    for resultado in resultados:
        ejercicio = resultado["ejercicio"]
        metricas = resultado["metricas"]
        principal = resultado["importancia"].iloc[0]["Variable"]
        lineas.append(
            f"| {ejercicio.nombre} | {resultado['filas_validas']} | "
            f"{metricas['MSE']:.4f} | {metricas['RMSE']:.4f} | "
            f"{metricas['R2']:.4f} | {principal} |"
        )

    for resultado in resultados:
        ejercicio = resultado["ejercicio"]
        modelo = resultado["modelo"]
        metricas = resultado["metricas"]
        lineas.extend(
            [
                "",
                f"## {ejercicio.nombre}",
                "",
                f"Objetivo: estimar `{ejercicio.objetivo}` en {ejercicio.unidad}.",
                "",
                f"Ecuación: `{ejercicio.objetivo} = {modelo.intercept_:.4f} "
                + " ".join(
                    f"{coef:+.4f} × {variable}"
                    for variable, coef in zip(ejercicio.variables, modelo.coef_)
                )
                + "`",
                "",
                f"MSE = {metricas['MSE']:.4f}; RMSE = {metricas['RMSE']:.4f}; "
                f"R² = {metricas['R2']:.4f}.",
                "",
                "Los coeficientes describen el cambio esperado en el objetivo al aumentar "
                "una unidad la variable correspondiente, manteniendo las demás constantes. "
                "La comparación de impacto se basa en magnitudes estandarizadas.",
                "",
                f"La variable con mayor impacto estandarizado en esta ejecución es "
                f"**{resultado['importancia'].iloc[0]['Variable']}**.",
                "",
                f"Gráficas: `graficas/{ejercicio.clave}_relaciones.png`, "
                f"`graficas/{ejercicio.clave}_correlaciones.png` y "
                f"`graficas/{ejercicio.clave}_real_estimado.png`.",
            ]
        )
    lineas.extend(
        [
            "",
            "## Limitaciones",
            "",
            "Los resultados describen estos conjuntos de datos y no garantizan el mismo "
            "desempeño con datos nuevos. En particular, hora y día de la semana son "
            "variables cíclicas que una regresión lineal directa representa de forma "
            "aproximada. La predicción de glucosa es un ejercicio académico y no tiene "
            "uso diagnóstico.",
            "",
        ]
    )
    (CARPETA_RESULTADOS / "informe_resultados.md").write_text(
        "\n".join(lineas), encoding="utf-8"
    )
