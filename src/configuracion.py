"""Definiciones de los escenarios que se entrenan y predicen."""

from dataclasses import dataclass
from pathlib import Path

RAIZ = Path(__file__).resolve().parents[1]
CARPETA_DATOS = RAIZ / "data"
CARPETA_MODELOS = RAIZ / "modelos"
CARPETA_GRAFICAS = RAIZ / "graficas"
CARPETA_RESULTADOS = RAIZ / "resultados"


@dataclass(frozen=True)
class Ejercicio:
    """Metadatos y variables de un problema de regresión."""

    clave: str
    nombre: str
    archivo_csv: str
    variables: tuple[str, ...]
    objetivo: str
    unidad: str
    descripcion: str


EJERCICIOS = (
    Ejercicio(
        clave="dolar",
        nombre="Precio del dólar",
        archivo_csv="dolar_data.csv",
        variables=("Dia", "Inflacion", "Tasa_interes"),
        objetivo="Precio_Dolar",
        unidad="pesos",
        descripcion="Estimar el precio del dólar a partir del día, la inflación y la tasa de interés.",
    ),
    Ejercicio(
        clave="glucosa",
        nombre="Nivel de glucosa",
        archivo_csv="glucosa_data.csv",
        variables=("Edad", "IMC", "Actividad_Fisica"),
        objetivo="Nivel_Glucosa",
        unidad="mg/dL",
        descripcion="Estimar glucosa según edad, índice de masa corporal y actividad física.",
    ),
    Ejercicio(
        clave="energia",
        nombre="Consumo de energía",
        archivo_csv="energia_data.csv",
        variables=("Temperatura", "Hora", "Dia_Semana"),
        objetivo="Consumo_Energia",
        unidad="kWh",
        descripcion="Estimar consumo eléctrico según temperatura, hora y día de la semana.",
    ),
)

EJERCICIOS_POR_CLAVE = {ejercicio.clave: ejercicio for ejercicio in EJERCICIOS}
