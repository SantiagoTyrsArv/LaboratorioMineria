"""Componentes y flujo de la interfaz de predicción."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import pandas as pd
import streamlit as st

from src.configuracion import CARPETA_MODELOS, EJERCICIOS, Ejercicio

CARPETA_PROYECTO = Path(__file__).resolve().parents[1]
ARCHIVO_ESTILOS = CARPETA_PROYECTO / "assets" / "interfaz.css"


@dataclass(frozen=True)
class CampoNumerico:
    """Parámetros de presentación y validación de un campo de entrada."""

    etiqueta: str
    minimo: int | float
    maximo: int | float
    inicial: int | float
    paso: int | float
    ayuda: str
    formato: str | None = None


CAMPOS: dict[str, CampoNumerico] = {
    "Dia": CampoNumerico("Día de la serie", 1, 1000, 250, 1, "Número de día usado por el modelo."),
    "Inflacion": CampoNumerico(
        "Inflación diaria", 0.0, 0.1, 0.02, 0.001, "Ingresa la tasa como decimal; por ejemplo, 0.02 equivale a 2 %.", "%.4f"
    ),
    "Tasa_interes": CampoNumerico(
        "Tasa de interés diaria (%)", 0.0, 20.0, 5.0, 0.1, "Tasa de interés en porcentaje."
    ),
    "Edad": CampoNumerico("Edad (años)", 1, 120, 45, 1, "Edad de la persona en años."),
    "IMC": CampoNumerico("Índice de masa corporal", 10.0, 60.0, 25.0, 0.1, "Valor de IMC."),
    "Actividad_Fisica": CampoNumerico(
        "Actividad física (horas/semana)", 0, 40, 4, 1, "Horas de actividad física a la semana."
    ),
    "Temperatura": CampoNumerico("Temperatura (°C)", -10.0, 50.0, 25.0, 0.5, "Temperatura ambiente."),
    "Hora": CampoNumerico("Hora del día", 1, 24, 12, 1, "Usa un número del 1 al 24."),
    "Dia_Semana": CampoNumerico(
        "Día de la semana", 1, 7, 1, 1, "1 corresponde al lunes y 7 al domingo."
    ),
}


@st.cache_resource
def cargar_modelo(clave: str) -> dict[str, Any]:
    """Carga el modelo seleccionado y conserva el objeto en caché."""
    return joblib.load(CARPETA_MODELOS / f"{clave}.joblib")


def configurar_pagina() -> None:
    """Configura la ventana y aplica el tema visual del proyecto."""
    st.set_page_config(
        page_title="Modelos de regresión | Laboratorio",
        page_icon="📊",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    estilos = ARCHIVO_ESTILOS.read_text(encoding="utf-8")
    st.markdown(f"<style>{estilos}</style>", unsafe_allow_html=True)


def dibujar_encabezado() -> None:
    """Muestra la cabecera que explica el propósito de la aplicación."""
    st.markdown(
        '<div class="brand-line"><span class="brand-symbol">r</span>'
        "Laboratorio de minería de datos</div>",
        unsafe_allow_html=True,
    )
    st.markdown(
        """
        <section class="hero">
          <div class="hero-copy">
            <div class="hero-tag">CRISP-DM / MODELOS PREDICTIVOS</div>
            <h1>De los datos<br>a una estimación.</h1>
            <p>Explora tres modelos de regresión lineal múltiple. Elige un escenario,
            ingresa los valores y observa cómo responde el modelo.</p>
          </div>
          <div class="hero-mark" aria-hidden="true">
            <span class="axis axis-x"></span><span class="axis axis-y"></span>
            <span class="plot-line"></span><span class="plot-point point-one"></span>
            <span class="plot-point point-two"></span><span class="plot-point point-three"></span>
            <span class="plot-point point-four"></span>
          </div>
        </section>
        """,
        unsafe_allow_html=True,
    )


def dibujar_selector() -> Ejercicio:
    """Muestra el selector de escenario y una descripción en la barra lateral."""
    with st.sidebar:
        st.markdown("## Escenario")
        seleccion = st.radio(
            "Selecciona un modelo",
            EJERCICIOS,
            format_func=lambda ejercicio: ejercicio.nombre,
            label_visibility="collapsed",
        )
        st.divider()
        st.caption("Modelo seleccionado")
        st.markdown(f"### {seleccion.nombre}")
        st.write(seleccion.descripcion)
        st.markdown(
            '<div class="sidebar-foot">Tres conjuntos de datos · Una interfaz de predicción</div>',
            unsafe_allow_html=True,
        )
    return seleccion


def dibujar_titulo_escenario(ejercicio: Ejercicio) -> None:
    """Presenta el objetivo y la descripción del escenario activo."""
    st.markdown(
        f'<div class="scenario-title"><span>Escenario activo</span><h2>{ejercicio.nombre}</h2>'
        f"<p>{ejercicio.descripcion}</p></div>",
        unsafe_allow_html=True,
    )


def dibujar_campos(ejercicio: Ejercicio) -> dict[str, int | float]:
    """Crea una entrada numérica para cada predictor del ejercicio."""
    valores: dict[str, int | float] = {}
    columnas = st.columns(len(ejercicio.variables), gap="medium")
    for columna, variable in zip(columnas, ejercicio.variables):
        campo = CAMPOS[variable]
        parametros: dict[str, Any] = {
            "label": campo.etiqueta,
            "min_value": campo.minimo,
            "max_value": campo.maximo,
            "value": campo.inicial,
            "step": campo.paso,
            "help": campo.ayuda,
            "key": f"{ejercicio.clave}_{variable}",
        }
        if campo.formato:
            parametros["format"] = campo.formato
        with columna:
            valores[variable] = st.number_input(**parametros)
    return valores


def dibujar_panel_entrada(ejercicio: Ejercicio) -> tuple[bool, dict[str, int | float]]:
    """Renderiza el formulario de datos y devuelve si se solicitó una predicción."""
    with st.container(border=True):
        st.markdown('<div class="panel-kicker">DATOS DE ENTRADA</div>', unsafe_allow_html=True)
        st.markdown("### Define los valores")
        st.caption("Introduce los datos que quieres evaluar con el modelo.")
        with st.form(f"formulario_{ejercicio.clave}"):
            valores = dibujar_campos(ejercicio)
            enviado = st.form_submit_button("Calcular estimación", use_container_width=True)
    return enviado, valores


def crear_prediccion(
    ejercicio: Ejercicio, paquete: dict[str, Any], valores: dict[str, int | float]
) -> float:
    """Construye la fila de entrada en el orden del modelo y predice el objetivo."""
    datos = [[valores[variable] for variable in paquete["variables"]]]
    entrada = pd.DataFrame(datos, columns=paquete["variables"])
    return float(paquete["modelo"].predict(entrada)[0])


def dibujar_panel_resultado(
    ejercicio: Ejercicio,
    paquete: dict[str, Any],
    estimacion: float | None,
) -> None:
    """Muestra la estimación o una instrucción inicial para usar el formulario."""
    with st.container(border=True):
        st.markdown('<div class="panel-kicker">RESULTADO</div>', unsafe_allow_html=True)
        st.markdown("### Predicción del modelo")
        if estimacion is None:
            st.markdown(
                '<div class="empty-result"><span class="empty-result-icon">↗</span>'
                "<strong>Aquí aparecerá la estimación</strong>"
                "<p>Completa los campos y calcula el resultado.</p></div>",
                unsafe_allow_html=True,
            )
            return

        st.metric(
            paquete["objetivo"],
            f"{estimacion:,.2f} {paquete['unidad']}",
        )
        st.caption("Resultado generado a partir de los valores ingresados.")
        if ejercicio.clave == "glucosa":
            st.info("Uso académico: esta estimación no es un diagnóstico médico.")


def dibujar_metricas(paquete: dict[str, Any]) -> None:
    """Presenta las métricas de desempeño calculadas sobre los datos de prueba."""
    st.markdown('<div class="section-rule"></div>', unsafe_allow_html=True)
    st.markdown("## Desempeño del modelo")
    st.caption("Métricas calculadas sobre observaciones que no participaron en el entrenamiento.")
    metricas = paquete["metricas"]
    columnas = st.columns(3, gap="medium")
    definiciones = (
        ("R²", "R2", "Variabilidad explicada por el modelo."),
        ("MSE", "MSE", "Promedio de los errores al cuadrado."),
        ("RMSE", "RMSE", "Error típico, en las unidades del resultado."),
    )
    for columna, (etiqueta, clave, ayuda) in zip(columnas, definiciones):
        with columna:
            st.metric(etiqueta, f"{metricas[clave]:,.4f}", help=ayuda)


def dibujar_detalles_modelo(paquete: dict[str, Any]) -> None:
    """Permite consultar la ecuación y el orden de impacto de los predictores."""
    modelo = paquete["modelo"]
    with st.expander("Ecuación e importancia de las variables"):
        terminos = " ".join(
            f"{coeficiente:+.4f} × {variable}"
            for variable, coeficiente in zip(paquete["variables"], modelo.coef_)
        )
        st.code(f"{paquete['objetivo']} = {modelo.intercept_:.4f} {terminos}")
        st.caption("La beta estandarizada permite comparar variables que tienen unidades distintas.")
        st.dataframe(
            pd.DataFrame(paquete["importancia"]),
            hide_index=True,
            use_container_width=True,
        )


def dibujar_modelo_no_disponible() -> None:
    """Explica cómo generar los archivos de modelo que necesita la aplicación."""
    st.warning("Todavía no se han generado los modelos.")
    st.write("Entrénalos desde la carpeta del proyecto para habilitar las predicciones.")
    st.code("python entrenar.py", language="bash")


def main() -> None:
    """Ejecuta el flujo completo de la interfaz."""
    configurar_pagina()
    dibujar_encabezado()
    ejercicio = dibujar_selector()
    dibujar_titulo_escenario(ejercicio)

    ruta = CARPETA_MODELOS / f"{ejercicio.clave}.joblib"
    if not ruta.is_file():
        dibujar_modelo_no_disponible()
        return

    paquete = cargar_modelo(ejercicio.clave)
    enviar, valores = dibujar_panel_entrada(ejercicio)
    clave_resultado = f"estimacion_{ejercicio.clave}"
    if enviar:
        st.session_state[clave_resultado] = crear_prediccion(ejercicio, paquete, valores)

    estimacion = st.session_state.get(clave_resultado)
    dibujar_panel_resultado(ejercicio, paquete, estimacion)
    dibujar_metricas(paquete)
    dibujar_detalles_modelo(paquete)
