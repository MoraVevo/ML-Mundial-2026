# ML Mundial 2026

Pipeline reproducible para recolectar datos de futbol, construir variables
prepartido y generar predicciones para el Mundial 2026.

El repositorio contiene datos estaticos necesarios para el calendario/resultados manuales,
scripts operativos y pruebas automatizadas. Los datos crudos, matrices generadas, modelos entrenados y
salidas de prediccion se mantienen fuera de Git.

Este repositorio se publica como demostracion tecnica y portafolio. El codigo,
la arquitectura del modelo, el diseño de features, reportes y assets son
propietarios; no se concede permiso para copiar, reutilizar, redistribuir o
explotar comercialmente el proyecto sin autorizacion previa.

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

Las credenciales de proveedores no se versionan. API-Football tiene una cuota
diaria limitada, por lo que la recoleccion completa esta pensada como un paso
operativo del mantenedor del proyecto. 
## Instalacion

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -e ".[dev]"
pytest
```

## Pipeline principal

El proyecto sigue una sola secuencia de trabajo:

1. Recolectar datos de proveedores y guardarlos en cache local.
2. Normalizar las fuentes a tablas consistentes.
3. Construir el frame de entrenamiento nacional.
4. Exportar la matriz limpia y la matriz neutral del modelo.
5. Entrenar el modelo neutral de Mundial.
6. Recalcular metricas y reporte tecnico.
7. Generar predicciones para los proximos partidos.

### 1. Recoleccion

```powershell
kinela collect fifa-ranking
kinela collect football-data-bulk
kinela collect api-football-world-cup-teams --last 15 --detail-limit 100
```

### 2. Limpieza y normalizacion

```powershell
kinela normalize api-football
kinela normalize football-data
kinela normalize fifa-ranking
```

### 3. Features y matrices

```powershell
kinela export training-frame-national
kinela export clean-training-matrix-national
kinela export neutral-training-matrix-national
```

### 4. Entrenamiento

```powershell
kinela train lightgbm-neutral
```

### 5. Metricas

```powershell
python scripts\update_worldcup2026_manual_detail_from_espn.py
python scripts\update_worldcup2026_manual_detail_from_thescore.py
python scripts\audit_worldcup2026_manual_detail_coverage.py
python scripts\worldcup2026_model_metrics.py
```

### 6. Prediccion

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

El repo mantiene fuera de Git los insumos pesados, credenciales y artefactos
generados. Por defecto no versiona:

- credenciales locales ni archivos `.env`
- respuestas crudas de proveedores
- matrices procesadas
- modelos entrenados
- predicciones, reportes y hojas generadas

## GitHub Actions

`CI` ejecuta lint y pruebas.

`Collect football data` es una automatizacion operativa para el mantenedor:
requiere secrets de proveedores, usa cache de respuestas y esta limitada por la
cuota diaria de las APIs. No es la via principal para que un usuario externo
pruebe el modelo; el orden reproducible del proyecto esta documentado en el
pipeline principal.

## Licencia

Copyright (c) 2026 MoraVevo. Todos los derechos reservados. Este proyecto se
publica para revision de portafolio y demostracion tecnica; cualquier uso,
copia, modificacion, redistribucion o explotacion comercial requiere permiso
previo por escrito.
