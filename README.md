# ML Mundial 2026

Pipeline reproducible para recolectar datos de futbol, construir variables
prepartido y generar predicciones para el Mundial 2026.

El repositorio contiene el modelo neutral del proyecto, datos estaticos
necesarios para el calendario/resultados manuales, scripts operativos y pruebas
automatizadas. Los datos crudos, matrices generadas, modelos entrenados y
salidas de prediccion se mantienen fuera de Git.

## Fuentes

| Fuente | Uso | Credencial |
|---|---|---|
| StatsBomb Open Data | Historial abierto de Mundial para contexto base | No |
| API-Football | Partidos recientes, estadisticas, lineups y jugadores | `APISPORTS_KEY` |
| football-data.org | Calendarios, resultados y contexto de competiciones | `FOOTBALL_DATA_TOKEN` |
| ESPN/theScore | Actualizacion puntual de resultados del Mundial 2026 | No |

Las respuestas originales se guardan en `data/raw/` y se reutilizan desde cache.
Los artefactos generados se escriben en `data/processed/`, `data/models/` y
`outputs/`; esas rutas estan ignoradas para mantener el repo liviano.

## Instalacion

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -e ".[dev]"
pytest
```

## Flujo principal

```powershell
kinela collect fifa-ranking
kinela collect football-data-bulk
kinela collect api-football-world-cup-teams --last 15 --detail-limit 100

kinela normalize api-football
kinela normalize football-data
kinela normalize fifa-ranking

kinela export training-frame-national
kinela export clean-training-matrix-national
kinela export neutral-training-matrix-national
kinela train lightgbm-neutral
```

Para actualizar resultados jugados del Mundial 2026 y recalcular metricas:

```powershell
python scripts\update_worldcup2026_manual_detail_from_espn.py
python scripts\update_worldcup2026_manual_detail_from_thescore.py
python scripts\audit_worldcup2026_manual_detail_coverage.py
python scripts\worldcup2026_default_auc_evaluation.py
```

Para generar predicciones de los proximos partidos:

```powershell
python scripts\predict_next4_with_all_played_worldcup.py --limit 4
```

## Evaluacion del modelo

El reporte tecnico con cortes temporales, metricas, matrices de confusion,
analisis de error e importancia de features esta en
[`docs/model_evaluation.md`](docs/model_evaluation.md).

## Modelo

El modelo de produccion es neutral y usa variables prepartido de ranking, forma
reciente, balance de goles, contexto de fase y compatibilidad tactica.

La evaluacion principal reporta accuracy sobre partidos ya jugados del Mundial
2026. Esos partidos se separan como test y no se usan para entrenar el modelo
que calcula ese accuracy. Para predicciones futuras, el script
`predict_next4_with_all_played_worldcup.py` entrena con todos los partidos
nacionales completados disponibles, incluyendo los resultados ya jugados del
Mundial.

## Estructura

```text
src/kinela/        Codigo del paquete y CLI
scripts/           Scripts operativos de actualizacion, auditoria y prediccion
tests/             Pruebas automatizadas
data/static/       Calendario, resultados manuales y configuracion estatica
.github/workflows/ CI y recoleccion programada
```

## Higiene del repositorio

Este repo esta preparado para poder hacerse publico despues de una revision
final. Por defecto no versiona:

- credenciales locales ni archivos `.env`
- respuestas crudas de proveedores
- matrices procesadas
- modelos entrenados
- predicciones, reportes y hojas generadas

## GitHub Actions

`CI` ejecuta lint y pruebas. `Collect football data` permite refrescar datos en
ejecuciones programadas o manuales y publica tablas procesadas como artifact
temporal.
