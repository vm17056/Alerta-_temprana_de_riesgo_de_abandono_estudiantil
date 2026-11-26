import pandas as pd

from src.riesgo_abandono import (
    CRITERIOS_PESO_2,
    CRITERIOS_PESO_3,
    PESOS_DIAGNOSTICO,
    diagnosticar_estudiantes,
    puntaje_checklist,
    validar_dataset_estudiantes,
)


def _fila_checklist(valor=True):
    fila = {c: valor for c in [*CRITERIOS_PESO_2, *CRITERIOS_PESO_3]}
    return fila


def test_ponderacion_oficial_suma_100():
    assert round(sum(PESOS_DIAGNOSTICO.values()), 10) == 1.0
    assert round(sum(PESOS_DIAGNOSTICO[c] for c in CRITERIOS_PESO_2), 10) == 0.5
    assert round(sum(PESOS_DIAGNOSTICO[c] for c in CRITERIOS_PESO_3), 10) == 0.3
    assert PESOS_DIAGNOSTICO["porcentaje_asistencia"] == 0.10
    assert PESOS_DIAGNOSTICO["promedio_notas"] == 0.10


def test_checklist_peso_2_y_peso_3():
    df = pd.DataFrame([{**_fila_checklist(False), "trabajo_infantil": True}, {**_fila_checklist(False), "bullying": True}])
    scores = puntaje_checklist(df)
    assert round(scores.iloc[0], 10) == round(0.50 / 7, 10)
    assert round(scores.iloc[1], 10) == 0.06


def test_diagnostico_completo_alcanza_100_por_ciento():
    fila = {
        "porcentaje_asistencia": 65,
        "promedio_notas": 3.5,
        **_fila_checklist(True),
    }
    resultado = diagnosticar_estudiantes(pd.DataFrame([fila]))
    assert resultado.loc[0, "score_riesgo_final"] == 1.0
    assert resultado.loc[0, "porcentaje_riesgo_final"] == 100.0
    assert resultado.loc[0, "cobertura_diagnostico"] == 1.0
    assert resultado.loc[0, "nivel_riesgo"] == "ALTO"


def test_diagnostico_sin_checklist_normaliza_sobre_criterios_disponibles():
    df = pd.DataFrame([{
        "porcentaje_asistencia": 65,
        "promedio_notas": 3.5,
    }])
    resultado = diagnosticar_estudiantes(df)
    assert resultado.loc[0, "riesgo_academico"] == 1.0
    assert resultado.loc[0, "cobertura_diagnostico"] == 0.20
    assert resultado.loc[0, "score_riesgo_final"] == 1.0
    assert resultado.loc[0, "nivel_riesgo"] == "ALTO"


def test_notas_0_100_se_normalizan():
    df = pd.DataFrame({"porcentaje_asistencia": [90], "promedio_notas": [80]})
    limpio = validar_dataset_estudiantes(df)
    assert limpio.loc[0, "promedio_notas"] == 8.0


def test_diagnosticar_estudiante_acepta_dict():
    from src.riesgo_abandono import diagnosticar_estudiante

    resultado = diagnosticar_estudiante({
        "porcentaje_asistencia": 80,
        "promedio_notas": 6.5,
        **_fila_checklist(False),
    })
    assert len(resultado) == 1
    assert resultado.loc[0, "nivel_riesgo"] in {"BAJO", "MEDIO", "ALTO"}


def test_campos_checklist_vacios_son_informacion_insuficiente():
    df = pd.DataFrame([{
        "porcentaje_asistencia": 80,
        "promedio_notas": 6.5,
        "trabajo_infantil": "",
        "migracion": "NO",
    }])
    resultado = diagnosticar_estudiantes(df)
    assert resultado.loc[0, "estado_informacion"] == "INFORMACION_INSUFICIENTE"
    assert "trabajo_infantil" in resultado.loc[0, "campos_sin_informacion"]
    assert resultado.loc[0, "nivel_riesgo"] in {"BAJO", "MEDIO", "ALTO"}


def test_campo_academico_vacio_no_inventa_riesgo():
    df = pd.DataFrame([{
        "porcentaje_asistencia": None,
        "promedio_notas": 6.5,
        "trabajo_infantil": "NO",
    }])
    resultado = diagnosticar_estudiantes(df)
    assert resultado.loc[0, "nivel_riesgo"] == "INFORMACION_INSUFICIENTE"
    assert pd.isna(resultado.loc[0, "score_riesgo_final"])
    assert resultado.loc[0, "requiere_completar_informacion"] is True or bool(resultado.loc[0, "requiere_completar_informacion"])
