import pandas as pd
from src.riesgo_abandono import diagnosticar_estudiantes, puntaje_checklist, validar_dataset_estudiantes


def test_checklist_mayor_peso_pesa_doble():
    df = pd.DataFrame([
        {"trabajo_infantil": 1, "vive_lejos": 0},
        {"trabajo_infantil": 0, "vive_lejos": 1},
    ])
    scores = puntaje_checklist(df)
    assert scores.iloc[0] == 0.2
    assert scores.iloc[1] == 0.1


def test_diagnostico_sin_modelo():
    df = pd.DataFrame([{
        "id_estudiante": "A1",
        "grado": 6,
        "porcentaje_asistencia": 65,
        "promedio_notas": 3.5,
        "trabajo_infantil": 1,
        "pobreza": 1,
        "bullying": 1,
    }])
    resultado = diagnosticar_estudiantes(df)
    assert resultado.loc[0, "riesgo_academico"] == 1.0
    assert resultado.loc[0, "nivel_riesgo"] == "ALTO"
    assert bool(resultado.loc[0, "requiere_revision"])


def test_notas_0_100_se_normalizan():
    df = pd.DataFrame({"porcentaje_asistencia": [90], "promedio_notas": [80]})
    limpio = validar_dataset_estudiantes(df)
    assert limpio.loc[0, "promedio_notas"] == 8.0


def test_diagnosticar_estudiante_acepta_dict():
    from src.riesgo_abandono import diagnosticar_estudiante

    resultado = diagnosticar_estudiante({
        "porcentaje_asistencia": 80,
        "promedio_notas": 6.5,
        "trabajo_infantil": True,
        "migracion": False,
        "embarazo_adolescente": False,
        "pobreza": True,
        "vive_lejos": False,
        "bullying": True,
    })
    assert len(resultado) == 1
    assert resultado.loc[0, "nivel_riesgo"] in {"BAJO", "MEDIO", "ALTO"}
