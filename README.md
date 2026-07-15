# ML Mundial 2026

Modelo de machine learning para prediccion de partidos del Mundial 2026. La
receta activa es un LightGBM neutral v10: conserva 9 variables prepartido para
los goles y combina por igual un clasificador base con otro que ve dos xG de
matchup. Su unica senal activa de fuerza es el FIFA SUM Live causal, reconstruido
con el ranking oficial vigente antes de cada partido y actualizado con la formula
publicada por FIFA. Logra 67/100 aciertos (`67.00%`) y log-loss `0.8688` en el
holdout del Mundial 2026 al 2026-07-11.

| Evaluacion | Accuracy | Correctos | Partidos | Lectura rapida |
|---|---:|---:|---:|---|
| Test externo temporal | 65.38% | 68/104 | 104 | Diagnostico externo en partidos oficiales recientes no vistos |
| Test Mundial 2026 | 67.00% | 67/100 | 100 | Holdout principal: partidos reales ya jugados del Mundial |
| Artefacto all-played | N/A | N/A | 887 train | Modelo para predicciones futuras con resultados jugados incorporados |

## Pronostico visual del torneo

La grafica principal usa cuatro paneles separados para mostrar las
probabilidades de terminar primero, segundo, tercero o cuarto. El flujo actual
de simulacion usa el artefacto all-played v10, con todos los partidos ya jugados
incorporados antes de predecir cruces pendientes, asignacion exacta de mejores
terceros y resolucion de empates en eliminatorias mediante prorroga verificada y,
si persiste el empate, penales.

![Probabilidades top 4 del Mundial 2026](docs/assets/worldcup_2026_top4_probabilities.png)

Pipeline reproducible para recolectar datos de futbol, construir variables
prepartido, entrenar el modelo y simular el torneo.

El repositorio contiene datos estaticos necesarios para el
calendario/resultados manuales, scripts operativos y pruebas automatizadas. Los
datos crudos, matrices generadas, modelos entrenados y salidas de prediccion se
mantienen fuera de Git.

Este repositorio se publica como demostracion tecnica y portafolio. El codigo,
la arquitectura del modelo, el diseno de features, reportes y assets son
propietarios; no se concede permiso para copiar, reutilizar, redistribuir o
explotar comercialmente el proyecto sin autorizacion previa.

## Fuentes

| Fuente | Uso | Credencial |
|---|---|---|
| StatsBomb Open Data | Historial de Mundial y torneos continentales; eventos de periodos 3/4 para prorroga | No |
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

## Nueva PC / setup portable

Para mover el estado actual del modelo a otra computadora sin volver a
recolectar datos, usar el flujo portable documentado en
[`docs/new_pc_setup.md`](docs/new_pc_setup.md). El repo via GitHub contiene el
codigo; el ZIP portable via Drive contiene `data/`, modelos entrenados y el
fixture necesario para simular.

## Pipeline principal

El proyecto sigue una sola secuencia de trabajo:

1. Recolectar datos de proveedores y guardarlos en cache local.
2. Normalizar las fuentes a tablas consistentes.
3. Construir el frame de entrenamiento nacional.
4. Exportar la matriz limpia y la matriz final del modelo.
5. Entrenar el modelo holdout del Mundial y el artefacto all-played para los cruces pendientes.
6. Recalcular metricas y reporte tecnico.
7. Simular el torneo y regenerar el visual publico.

### 1. Recoleccion

```powershell
kinela collect fifa-ranking
kinela collect fifa-ranking-history --from-date 2022-01-01
kinela collect football-data-bulk
kinela collect api-football-world-cup-teams --last 15 --detail-limit 100
python scripts\collect_penalty_shootout_data.py --refresh
python scripts\collect_extra_time_data.py
```

### 2. Limpieza y normalizacion

```powershell
kinela normalize api-football
kinela normalize football-data
kinela normalize fifa-ranking
kinela normalize fifa-ranking-history
```

### 3. Features y matrices

```powershell
kinela export training-frame-national
kinela export clean-training-matrix-national
kinela export neutral-training-matrix-national
```

### 4. Entrenamiento

```powershell
python scripts\train_worldcup2026_holdout_model.py
python scripts\train_worldcup2026_all_played_model.py
python scripts\train_penalty_shootout_model.py
python scripts\train_extra_time_model.py
```

### 5. Metricas

```powershell
python scripts\update_worldcup2026_manual_detail_from_espn.py
python scripts\update_worldcup2026_manual_detail_from_thescore.py
python scripts\audit_worldcup2026_manual_detail_coverage.py
python scripts\worldcup2026_model_metrics.py
python scripts\generate_model_evaluation_report.py
python scripts\worldcup2026_rating_signal_ablation.py
```

### 6. Simulacion del torneo

```powershell
python scripts\run_worldcup2026_consensus_bracket.py --runs 5000 --workers 8 --seed 42 --progress-every 25 --model-path data\models\lightgbm_neutral_all_played_wc2026.joblib --model-label all_played_wc2026_v10_fifa_sum_live_honest_5000 --output outputs\worldcup2026_consensus_bracket_5000_v10.json
python scripts\generate_worldcup2026_top4_visual.py --input outputs\worldcup2026_consensus_bracket_5000_v10.json
```

## Evaluacion del modelo

El reporte tecnico con cortes temporales, metricas, matrices de confusion,
analisis de error e importancia de features esta en
[`docs/model_evaluation.md`](docs/model_evaluation.md).

## Modelo

El modelo de produccion usa variables prepartido de ranking, forma reciente,
balance de goles, contexto de fase, score timing validado, historia actual del
Mundial y ventaja de finalizador diferencial. Un clasificador paralelo recibe
ademas el xG que cada lado suele crear combinado con el xG que su rival suele
conceder; sus probabilidades se mezclan 50/50 con las del clasificador base.
Los regresores de goles no reciben este xG porque en validacion agregaba ruido
al MAE.

La revision v10 comparo por separado FIFA oficial, FIFA SUM Live reconstruido y
el Elo interno. FIFA oficial y FIFA Live se solapan casi por completo
(`r=0.99527` en el holdout actual), asi que no se activan simultaneamente. El
modelo conserva solo el
FIFA SUM Live: parte de cada publicacion oficial y actualiza causalmente despues
de cada partido con importancia, resultado de 90/prorroga/penales y proteccion
eliminatoria.
El Elo interno sigue disponible como diagnostico, pero no alimenta ninguna
feature activa; sus parametros se habian seleccionado mirando el mismo holdout y
su mejora aparente no era evidencia suficientemente honesta para produccion.

La prorroga se modela aparte de los 90 minutos. El dataset usa exclusivamente
eventos abiertos de StatsBomb con periodos 3 y 4 verificados (29 partidos de
Mundial, Euro, AFCON y Copa America; `Data provided by StatsBomb`). La tasa
seleccionada es 0.621 goles adicionales por prorroga. El reparto por fuerza/xG
solo mejoro 0.2% el NLL fuera de torneo y su intervalo bootstrap incluyo
empeoramiento, por lo que produccion conserva un reparto neutral. Los goles de
prorroga se guardan en campos separados y nunca entran al historial ni a las
features de 90 minutos. Si persiste el empate, se aplica el modelo de penales
seleccionado (50/50).

La evaluacion principal reporta accuracy sobre partidos ya jugados del Mundial
2026. Esos partidos se separan como test y no se usan para entrenar el modelo
que calcula ese accuracy. Las predicciones futuras y simulaciones largas usan
el artefacto all-played, que incorpora todos los resultados ya jugados antes de
predecir cruces pendientes.

El modelo de tandas v2 usa los datos CC0 de
[`martj42/international_results`](https://github.com/martj42/international_results),
filtra equipos no afiliados a FIFA antes de construir historiales y compara
regresion logistica simetrica, random forest, gradient boosting y 50/50 en
cortes temporales. La seleccion pre-2018 favorece 50/50 por log-loss y Brier;
por eso produccion ya no usa la diferencia de calidad de plantilla del modelo
anterior. En el backtest moderno 2018-2025 la accuracy esperada es 50.00%
(133 tandas, intervalo 95% 41.62%-58.38%). El orden del primer tirador se audita
por separado porque solo se conoce despues del sorteo y no mejoro el log-loss
moderno. El reporte reproducible se genera en
`outputs/penalty_shootout_model_evaluation.json`.

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
