# Nueva PC: setup portable del modelo Mundial 2026

Este flujo deja la nueva computadora lista para simular sin volver a recolectar
datos de proveedores. La idea recomendada es:

- GitHub: codigo, scripts, tests y configuracion.
- Google Drive u otro cloud: bundle ZIP con `data/` y modelos entrenados.

El ZIP portable no incluye el codigo del proyecto. Primero se debe clonar o
actualizar el repo desde GitHub; despues se restaura el ZIP encima para traer
los datos y modelos.

No hace falta crear otro repositorio. Conviene mantener un solo repo y mover los
artefactos pesados fuera de Git.

## 1. En la PC actual: crear el bundle

Desde la carpeta del proyecto:

```cmd
python scripts\create_portable_bundle.py --mode simulation
```

Esto crea un ZIP en `outputs\portable\`. El modo `simulation` incluye lo
necesario para correr simulaciones:

- `data/static`
- `data/processed`
- modelos principales en `data/models`
- subcache de lineups/jugadores en `data/raw/api_football/fixtures`
- subcache de perfiles de jugadores en `data/raw/api_football/players`
- cache ESPN del Mundial en `data/raw/espn/worldcup_2026`
- fixture del Mundial en `data/raw/football_data/competitions/WC`
- metadata del modelo all-played si existe

Ese ZIP deberia ser mucho mas liviano que copiar todo `data/raw`.

Si se quiere mover absolutamente todo el cache crudo para poder renormalizar sin
recolectar APIs, usar:

```cmd
python scripts\create_portable_bundle.py --mode full-cache
```

Luego subir el ZIP a Drive.

## 2. En la nueva PC: clonar e instalar

Instalar primero Git y Python 3.11 o 3.12. Luego, en `cmd`:

```cmd
git clone https://github.com/MoraVevo/ML-Mundial-2026.git
cd ML-Mundial-2026
py -3.12 -m venv .venv
.venv\Scripts\python -m pip install --upgrade pip
.venv\Scripts\python -m pip install -e ".[dev]"
```

Si `py -3.12` no existe, usar:

```cmd
python -m venv .venv
```

## 3. Restaurar el bundle

Descargar el ZIP desde Drive y correr:

```cmd
.venv\Scripts\python scripts\restore_portable_bundle.py --bundle "C:\ruta\al\kinela_worldcup2026_simulation_bundle_2026-07-04.zip" --overwrite
```

## 4. Verificar que todo carga

```cmd
.venv\Scripts\python scripts\verify_portable_setup.py
```

Para una prueba mas fuerte, que ademas corre una simulacion completa:

```cmd
.venv\Scripts\python scripts\verify_portable_setup.py --quick-sim
```

## 5. Correr la simulacion larga

```cmd
.venv\Scripts\python scripts\run_worldcup2026_consensus_bracket.py --runs 5000 --workers 8 --seed 42 --progress-every 25 --model-path data\models\lightgbm_neutral_all_played_wc2026.joblib --model-label all_played_wc2026_v7_full_context_5000 --output outputs\worldcup2026_consensus_bracket_5000_nueva_pc.json
```

El script imprime progreso cada 25 iteraciones con los tres campeones mas
frecuentes y su porcentaje.

## Cuando si harian falta credenciales

Para solo simular con el bundle, no hacen falta API keys.

Las credenciales solo hacen falta si en la nueva PC se quiere actualizar o
recolectar datos nuevos:

- `APISPORTS_KEY`
- `FOOTBALL_DATA_TOKEN`

En ese caso se puede copiar `.env.example` a `.env` y llenar las credenciales.
