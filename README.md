# Kinela

Pipeline reproducible para recolectar datos históricos de fútbol, generar
variables predictivas y entrenar modelos de goles y resultados.

## Fuentes

| Fuente | Uso | Credencial |
|---|---|---|
| StatsBomb Open Data | Mundial 2022: partidos, alineaciones y eventos | No |
| API-Football | Últimos partidos, eventos, estadísticas de equipo y jugadores | `APISPORTS_KEY` |
| football-data.org | Resultados, plantillas y agregados por jugador | `FOOTBALL_DATA_TOKEN` |

Las respuestas originales se guardan bajo `data/raw/`. Si el archivo ya existe,
el recolector lo reutiliza y no consume otra llamada. Los archivos descargados
no se versionan porque pueden ser grandes o estar sujetos a condiciones del
proveedor.

## Inicio rápido

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -e ".[dev]"
kinela collect statsbomb-world-cup-2022
kinela collect fifa-ranking
kinela normalize statsbomb-world-cup-2022
kinela normalize api-football
kinela normalize football-data
kinela normalize fifa-ranking
kinela export clean-training-matrix
kinela export neutral-training-matrix
kinela train lightgbm-neutral
pytest
```

Los resultados normalizados se escriben en:

```text
data/processed/statsbomb_world_cup_2022/
├── goals.csv
├── lineups.csv
├── matches.csv
├── player_match_stats.csv
└── team_match_stats.csv
```

`possession_event_share_pct` es una aproximación basada en la proporción de
eventos de cada equipo, no el dato oficial de tiempo de posesión. Los goles
desde campo propio se detectan por coordenada de inicio. StatsBomb no etiqueta
un tipo separado llamado "media cancha". Las tandas se conservan en
`goals.csv` con `is_shootout=true`, pero no se suman como goles del partido.

## Credenciales

Crear las cuentas de los dos proveedores y definir las variables:

```powershell
$env:APISPORTS_KEY="..."
$env:FOOTBALL_DATA_TOKEN="..."
```

Nunca se deben guardar claves en Git. Para GitHub Actions se configuran como
repository secrets con los mismos nombres.

Ejemplos de descarga en caché:

```powershell
# Una llamada para los últimos 15 partidos.
kinela collect api-football-team --team-id 26 --last 15

# Quince llamadas adicionales, solo la primera vez, para eventos, alineaciones,
# estadísticas de equipo y estadísticas individuales de cada partido.
kinela collect api-football-team --team-id 26 --last 15 --details

kinela collect football-data-team --team-id 760 --last 15

# Últimos 15 resultados de las selecciones del Mundial actual.
kinela collect football-data-world-cup-teams --last 15

# Partidos, equipos y goleadores de todas las competiciones accesibles.
kinela collect football-data-bulk
```

Para las 32 selecciones del Mundial 2022, el siguiente comando descarga una
vez el catálogo, consulta las temporadas 2023 y 2024, guarda los últimos 15
partidos disponibles de cada selección y completa un lote de detalles:

```powershell
kinela collect api-football-world-cup-teams --last 15 --detail-limit 25
```

Al repetirlo otro día, los archivos existentes no generan llamadas. El resumen
indica cuántos detalles quedan pendientes. No usar `--refresh` salvo que se
quiera reemplazar deliberadamente información ya guardada.

El plan gratuito de API-Football no permite el parámetro `last` ni temporadas
posteriores a 2024. Por eso el pipeline consulta temporadas históricas
permitidas y hace el ordenamiento y recorte localmente.

## Cobertura y cuotas

API-Football ofrece un plan gratuito de 100 llamadas por día. Obtener el último
partido de cada selección y luego el detalle de cada encuentro puede superar
esa cuota, por lo que la recolección debe hacerse por etapas y conservar el
caché. La disponibilidad de estadísticas depende de la cobertura declarada por
cada competición y temporada.

Publicaciones que utilicen StatsBomb Open Data deben atribuir la fuente a
StatsBomb y respetar sus condiciones. football-data.org también exige
atribución visible.

## GitHub Actions

`CI` ejecuta lint y pruebas en cada cambio. `Collect football data` reconstruye
el Mundial 2022 semanalmente, reutiliza el caché de descargas y publica las
tablas procesadas como artifact durante 30 días. Desde `workflow_dispatch`
también acepta IDs de equipo para los proveedores con credenciales.

## Primer modelo

El baseline usa partidos históricos ya cacheados de API-Football. El ETL crea:

```text
data/processed/api_football/
├── baseline_predictions.csv
├── matches.csv
└── team_match_features.csv
```

`team_match_features.csv` distingue amistosos, eliminatorias y torneos mayores,
conserva la ronda original del proveedor y calcula días de descanso desde el
partido anterior de cada selección. El modelo inicial predice goles esperados
con una distribución Poisson basada en promedios de equipo y devuelve
probabilidades de local, empate y visitante.

## Nota vigente: solo neutral

Para el proyecto de Mundial, el flujo soportado es neutral-only. No se debe
entrenar ni usar un modelo de local/visitante. Si una tabla cruda contiene
`home`/`away`, esos nombres vienen del esquema del proveedor y deben
transformarse a `team_a`/`team_b` antes del entrenamiento.
