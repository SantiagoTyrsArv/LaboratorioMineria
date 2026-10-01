"""Punto de entrada para entrenar y exportar los tres modelos."""

from __future__ import annotations

from src.entrenamiento import entrenar_todos


def main() -> None:
    """Entrena escenarios e imprime métricas y coeficientes."""
    for resultado in entrenar_todos():
        ejercicio = resultado["ejercicio"]
        modelo = resultado["modelo"]
        print(f"\n{ejercicio.nombre} | {resultado['filas_validas']} filas válidas")
        print(f"Intercepto: {modelo.intercept_:.4f}")
        for variable, coeficiente in zip(ejercicio.variables, modelo.coef_):
            sentido = "aumenta" if coeficiente >= 0 else "disminuye"
            print(f"{variable}: {coeficiente:+.4f} ({sentido} {ejercicio.objetivo} por unidad)")
        print("Métricas:", ", ".join(f"{k}={v:.4f}" for k, v in resultado["metricas"].items()))
        print("Importancia estandarizada:")
        print(resultado["importancia"].round(4).to_string(index=False))
        print(f"Gráficas y modelo guardados para '{ejercicio.clave}'.")


if __name__ == "__main__":
    main()
