import pandas as pd

from src.batch_analysis import leer_archivo_tabular, resumen_diagnostico
from src.riesgo_abandono import diagnosticar_estudiantes


def test_excel_aliases_and_batch_without_checklist(tmp_path):
    archivo = tmp_path / "salon.xlsx"
    pd.DataFrame(
        {
            "ID Estudiante": ["E001", "E002"],
            "Grado": [6, 7],
            "Asistencia": [95, 65],
            "Promedio": [8.5, 4.0],
        }
    ).to_excel(archivo, index=False)

    df = leer_archivo_tabular(archivo)
    resultado = diagnosticar_estudiantes(df)

    assert "porcentaje_asistencia" in df.columns
    assert "promedio_notas" in df.columns
    assert (resultado["estado_checklist"] == "NO INFORMADO").all()
    assert resultado.loc[1, "nivel_riesgo"] == "ALTO"


def test_resumen_batch():
    df = pd.DataFrame(
        {
            "porcentaje_asistencia": [95, 65],
            "promedio_notas": [8.5, 4.0],
        }
    )
    resultado = diagnosticar_estudiantes(df)
    resumen = resumen_diagnostico(resultado)

    total = resumen.loc[resumen["indicador"] == "Total de estudiantes", "valor"].iloc[0]
    alto = resumen.loc[resumen["indicador"] == "Riesgo alto", "valor"].iloc[0]

    assert total == 2
    assert alto == 1
