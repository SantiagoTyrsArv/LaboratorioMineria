"""Interfaz Streamlit para consultar los modelos entrenados."""

from __future__ import annotations

import joblib
import pandas as pd
import streamlit as st

from src.configuracion import CARPETA_MODELOS, EJERCICIOS, Ejercicio


st.set_page_config(
    page_title="Laboratorio | Modelos predictivos",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=IBM+Plex+Mono:wght@400;500&display=swap');

    :root {
        --ink: #17343b;
        --muted: #61767a;
        --paper: #f1f5f2;
        --panel: #ffffff;
        --line: #dce6e1;
        --teal: #145d5a;
        --lime: #d8ef72;
        --focus: #187b75;
    }

    .stApp {
        background: var(--paper);
        color: var(--ink);
        font-family: 'DM Sans', sans-serif;
    }
    .block-container {
        max-width: 1220px;
        padding-top: 2.2rem;
        padding-bottom: 4rem;
    }
    [data-testid="stSidebar"] {
        background: #e5ede8;
        border-right: 1px solid var(--line);
    }
    [data-testid="stSidebar"] > div:first-child { padding-top: 2rem; }
    h1, h2, h3, p, label, li { color: var(--ink); }
    h1 {
        font-size: clamp(2rem, 4vw, 3.3rem) !important;
        line-height: 1.05 !important;
        letter-spacing: -0.055em !important;
        font-weight: 600 !important;
    }
    h2, h3 { letter-spacing: -0.035em; }
    [data-testid="stCaptionContainer"] p { color: var(--muted); }

    .brand-mark {
        display: inline-flex;
        align-items: center;
        gap: 0.55rem;
        color: var(--teal);
        font-size: 0.82rem;
        font-weight: 700;
        letter-spacing: 0.015em;
        margin-bottom: 0.65rem;
    }
    .brand-dot {
        width: 0.65rem;
        height: 0.65rem;
        border-radius: 50%;
        background: var(--teal);
        box-shadow: 0 0 0 4px #c9ded2;
    }
    .hero {
        position: relative;
        overflow: hidden;
        padding: clamp(1.5rem, 4vw, 3rem);
        margin: 0.5rem 0 1.7rem;
        border-radius: 7px 28px 7px 28px;
        color: #f7fbf8;
        background: var(--teal);
        border-bottom: 5px solid var(--lime);
    }
    .hero::after {
        content: '';
        position: absolute;
        width: 250px;
        height: 250px;
        right: 5%;
        top: -135px;
        border: 1px solid rgba(216, 239, 114, 0.38);
        border-radius: 50%;
        box-shadow: 0 0 0 28px rgba(216, 239, 114, 0.06),
                    0 0 0 58px rgba(216, 239, 114, 0.045);
    }
    .hero-kicker {
        color: var(--lime);
        font-size: 0.8rem;
        font-weight: 600;
        margin-bottom: 0.8rem;
    }
    .hero-title {
        color: #f7fbf8;
        font-size: clamp(1.8rem, 4vw, 3rem);
        font-weight: 600;
        letter-spacing: -0.055em;
        line-height: 1.08;
        max-width: 680px;
    }
    .hero-copy {
        color: #d9e8e0;
        font-size: 1rem;
        line-height: 1.6;
        max-width: 610px;
        margin-top: 0.9rem;
    }
    .section-heading {
        font-size: 1.35rem;
        font-weight: 600;
        letter-spacing: -0.035em;
        margin: 0.35rem 0 0.2rem;
    }
    .section-copy { color: var(--muted); margin-bottom: 1rem; }
    .panel-note {
        border-left: 3px solid var(--lime);
        padding: 0.85rem 1rem;
        background: #f7f9ef;
        border-radius: 0 8px 8px 0;
        color: #42585b;
        font-size: 0.92rem;
        line-height: 1.55;
    }
    .result-placeholder {
        min-height: 205px;
        display: flex;
        flex-direction: column;
        justify-content: center;
        padding: 1.4rem;
        border: 1px dashed #b7c8bf;
        border-radius: 14px;
        background: #f7faf7;
    }
    .result-placeholder strong { font-size: 1.15rem; color: var(--ink); }
    .result-placeholder span { color: var(--muted); margin-top: 0.45rem; line-height: 1.5; }
    .stRadio [role="radiogroup"] { gap: 0.6rem; }
    .stRadio [role="radio"] {
        min-height: 2.75rem;
        padding: 0.45rem 0.9rem;
        border: 1px solid var(--line);
        border-radius: 999px;
        background: rgba(255,255,255,0.72);
    }
    .stRadio [role="radio"]:focus-visible,
    .stButton button:focus-visible,
    input:focus-visible {
        outline: 3px solid #8aac37 !important;
        outline-offset: 2px;
    }
    .stButton button, [data-testid="stFormSubmitButton"] button {
        border: 0;
        border-radius: 999px;
        background: var(--teal);
        color: white;
        font-weight: 600;
        padding: 0.65rem 1.2rem;
        min-height: 2.8rem;
    }
    .stButton button:hover, [data-testid="stFormSubmitButton"] button:hover {
        color: var(--ink);
        background: var(--lime);
        border: 0;
    }
    [data-testid="stMetric"] {
        padding: 0.9rem 1rem;
        border: 1px solid var(--line);
        border-radius: 12px;
        background: var(--panel);
    }
    [data-testid="stMetricLabel"] p { color: var(--muted); }
    [data-testid="stMetricValue"] { color: var(--teal); }
    [data-testid="stDataFrame"] { border: 1px solid var(--line); border-radius: 10px; }
    code, pre { font-family: 'IBM Plex Mono', monospace !important; }
    @media (max-width: 700px) {
        .block-container { padding: 1.2rem 1rem 3rem; }
        .hero { border-radius: 6px 20px 6px 20px; }
        .stRadio [role="radiogroup"] { flex-wrap: wrap; }
    }
    @media (prefers-reduced-motion: reduce) {
        *, *::before, *::after { scroll-behavior: auto !important; transition: none !important; }
    }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource
def cargar_modelo(clave: str) -> dict:
    """Carga y conserva en caché el paquete del modelo seleccionado."""
    return joblib.load(CARPETA_MODELOS / f"{clave}.joblib")


def construir_entrada(ejercicio: Ejercicio) -> dict[str, float]:
    """Dibuja controles con rangos ajustados al significado de cada variable."""
    opciones = {
        "Dia": ("Día de la serie", 1, 1000, 250, 1, None),
        "Inflacion": ("Inflación diaria", 0.0, 0.1, 0.02, 0.001, "%.4f"),
        "Tasa_interes": ("Tasa de interés diaria (%)", 0.0, 20.0, 5.0, 0.1, None),
        "Edad": ("Edad (años)", 1, 120, 45, 1, None),
        "IMC": ("Índice de masa corporal", 10.0, 60.0, 25.0, 0.1, None),
        "Actividad_Fisica": ("Actividad física (horas/semana)", 0, 40, 4, 1, None),
        "Temperatura": ("Temperatura (°C)", -10.0, 50.0, 25.0, 0.5, None),
        "Hora": ("Hora del día (1 a 24)", 1, 24, 12, 1, None),
        "Dia_Semana": ("Día de la semana (1=lunes, 7=domingo)", 1, 7, 1, 1, None),
    }
    valores: dict[str, float] = {}
    columnas = st.columns(2)
    for indice, variable in enumerate(ejercicio.variables):
        etiqueta, minimo, maximo, inicial, paso, formato = opciones[variable]
        parametros = {
            "label": etiqueta,
            "min_value": minimo,
            "max_value": maximo,
            "value": inicial,
            "step": paso,
            "key": f"{ejercicio.clave}_{variable}",
        }
        if formato:
            parametros["format"] = formato
        with columnas[indice % len(columnas)]:
            valores[variable] = st.number_input(**parametros)
    return valores


st.markdown(
    '<div class="brand-mark"><span class="brand-dot"></span>Laboratorio de minería de datos</div>',
    unsafe_allow_html=True,
)
st.markdown(
    """
    <section class="hero">
      <div class="hero-kicker">Modelos predictivos · CRISP-DM</div>
      <div class="hero-title">Explora los datos.<br>Interpreta cada predicción.</div>
      <div class="hero-copy">Tres ejercicios de regresión lineal múltiple para relacionar
      variables cotidianas con resultados que puedes estimar e interpretar.</div>
    </section>
    """,
    unsafe_allow_html=True,
)

with st.sidebar:
    st.markdown("### Mesa de análisis")
    st.caption("Elige un conjunto de variables para abrir su modelo.")

seleccion = st.radio(
    "Escenario de análisis",
    EJERCICIOS,
    format_func=lambda ejercicio: ejercicio.nombre,
    horizontal=True,
    label_visibility="collapsed",
)
st.markdown(f'<div class="section-heading">{seleccion.nombre}</div>', unsafe_allow_html=True)
st.markdown(f'<div class="section-copy">{seleccion.descripcion}</div>', unsafe_allow_html=True)

ruta_modelo = CARPETA_MODELOS / f"{seleccion.clave}.joblib"
if not ruta_modelo.is_file():
    st.info("Entrena los modelos para habilitar las predicciones.")
    st.code("python entrenar.py", language="bash")
    st.stop()

paquete = cargar_modelo(seleccion.clave)
modelo = paquete["modelo"]
col_entrada, col_resultado = st.columns([1.08, 0.92], gap="large")

with col_entrada:
    with st.container(border=True):
        st.markdown("#### Variables de entrada")
        st.caption("Ajusta los valores y calcula una estimación con el modelo entrenado.")
        with st.form(f"prediccion_{seleccion.clave}"):
            valores = construir_entrada(seleccion)
            enviar = st.form_submit_button("Calcular predicción", use_container_width=True)

with col_resultado:
    with st.container(border=True):
        st.markdown("#### Resultado")
        if enviar:
            entrada = pd.DataFrame(
                [[valores[variable] for variable in paquete["variables"]]],
                columns=paquete["variables"],
            )
            estimacion = float(modelo.predict(entrada)[0])
            st.metric(
                f"{paquete['objetivo']} estimado",
                f"{estimacion:,.2f} {paquete['unidad']}",
            )
            st.caption("Estimación calculada con los valores ingresados.")
            if seleccion.clave == "glucosa":
                st.markdown(
                    '<div class="panel-note">Referencia académica en ayunas. '
                    "Esta estimación no constituye un diagnóstico médico.</div>",
                    unsafe_allow_html=True,
                )
        else:
            st.markdown(
                '<div class="result-placeholder"><strong>Tu estimación aparecerá aquí</strong>'
                "<span>Completa las variables y pulsa «Calcular predicción» para consultar el resultado.</span></div>",
                unsafe_allow_html=True,
            )

st.divider()
st.markdown("#### Cómo se comporta el modelo")
st.caption("Indicadores calculados sobre el conjunto de prueba; una variable con mayor beta tiene más impacto relativo.")
metrica = paquete["metricas"]
col_r2, col_mse, col_rmse = st.columns(3)
col_r2.metric("R²", f"{metrica['R2']:.4f}", help="Proporción de variabilidad explicada en el conjunto de prueba.")
col_mse.metric("MSE", f"{metrica['MSE']:,.2f}", help="Promedio de errores al cuadrado.")
col_rmse.metric("RMSE", f"{metrica['RMSE']:,.2f}", help="Error típico expresado en las unidades del objetivo.")

with st.expander("Ver ecuación y peso de cada variable"):
    ecuacion = " ".join(
        f"{coeficiente:+.4f} × {variable}"
        for variable, coeficiente in zip(paquete["variables"], modelo.coef_)
    )
    st.code(f"{paquete['objetivo']} = {modelo.intercept_:.4f} {ecuacion}")
    st.dataframe(
        pd.DataFrame(paquete["importancia"]),
        hide_index=True,
        use_container_width=True,
    )
