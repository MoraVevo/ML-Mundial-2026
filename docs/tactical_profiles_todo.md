# Tactical Profile Features

Objetivo: mejorar cobertura de API-Football y probar perfiles tacticos
agregados, no estadisticas crudas sueltas. Los perfiles raw siguen fuera del
modelo default; solo se activa una senal tratada si mejora metricas y encaja con
la receta parsimoniosa.

## Estado actual

- API-Football non-friendly fixtures: 967
- Detalles validos actuales: 660
- Cobertura valida actual: 68.3%
- Detalles faltantes: 307
- La prueba con perfiles/raw tacticos completos siguio sin ser suficiente para
  activar un paquete amplio.
- La primera receta parsimoniosa uso `tactical_profile_edge` como una senal
  tratada dentro de 10 features totales y mejoro el default66:
  - default split: accuracy `0.6917 -> 0.7293`, log loss `0.7876 -> 0.7581`,
    MAE sum `2.3365 -> 2.1868`.
- Despues de agregar manualmente Suecia 5-1 Tunez, una nueva busqueda encontro
  un default mejor sin tactica: `parsimonious_guardrail_no_tactical`. La receta
  final remueve `tactical_profile_edge` y agrega `rating_guardrail_edge`.
  Metricas contra la receta tactica post-Suecia/Tunez:
  - default split: accuracy `0.7293 -> 0.7444`, log loss `0.7675 -> 0.7593`,
    MAE sum `2.2240 -> 2.2021`;
  - multi-window mean: accuracy `0.6867 -> 0.6868`, log loss `0.8073 -> 0.8028`,
    MAE sum `2.2016 -> 2.1734`.

## Descarga pendiente

Cuando resetee la cuota diaria de API-Football:

```powershell
.\.venv\Scripts\python.exe -m kinela.cli collect api-football-missing-details --detail-limit 307
.\.venv\Scripts\python.exe -m kinela.cli normalize api-football
.\.venv\Scripts\python.exe -m kinela.cli train lightgbm-neutral
```

Despues de normalizar, medir cobertura antes de ampliar perfiles. La cobertura
ya paso la meta minima de 60%, pero los paquetes raw siguen siendo ruidosos; la
ruta preferida es seguir con senales tratadas pequenas y auditables.

## Perfiles a probar

### tactical_attacking_pressure_diff

Perfil ofensivo: mide volumen y calidad basica de llegada.

Componentes:

- `recent6_shots_on_goal_diff`
- `recent6_shots_inside_box_diff`
- `recent6_total_shots_diff`
- `recent6_corner_kicks_diff`

Formula inicial probada:

```text
1.00 * shots_on_goal
+ 0.35 * shots_inside_box
+ 0.12 * total_shots
+ 0.25 * corner_kicks
```

### tactical_buildup_control_diff

Perfil de construccion: mide control con posesion y pases.

Componentes:

- `recent6_ball_possession_pct_diff`
- `recent6_passes_accurate_diff`
- `recent6_passes_pct_diff`

Formula inicial probada:

```text
0.10 * ball_possession_pct
+ 0.006 * passes_accurate
+ 0.05 * passes_pct
```

### tactical_direct_play_diff

Perfil directo/contraataque: busca equipos verticales, con tiros y rupturas,
pero menos dependencia de posesion/pases.

Componentes:

- `recent6_total_shots_diff`
- `recent6_offsides_diff`
- `recent6_shots_outside_box_diff`
- `recent6_total_passes_diff`
- `recent6_ball_possession_pct_diff`

Formula inicial probada:

```text
0.30 * total_shots
+ 0.45 * offsides
+ 0.25 * shots_outside_box
- 0.0025 * total_passes
- 0.025 * ball_possession_pct
```

### tactical_defensive_stress_diff

Perfil de estres defensivo/disciplina: no es fuerza defensiva pura; mide si un
equipo viene defendiendo bajo presion o jugando partidos mas sucios.

Componentes:

- `recent6_goalkeeper_saves_diff`
- `recent6_fouls_diff`
- `recent6_yellow_cards_diff`

Formula inicial probada:

```text
0.80 * goalkeeper_saves
+ 0.20 * fouls
+ 0.65 * yellow_cards
```

## Proximos perfiles candidatos

- `tactical_set_piece_pressure_diff`: corners, faltas recibidas si se consigue,
  goles/tiros de pelota parada si el proveedor lo permite.
- `tactical_finishing_profile_diff`: goles por tiro a puerta, con cuidado de no
  meter ruido por muestras pequenas.
- `tactical_allowed_pressure_diff`: tiros/corners recibidos, derivados desde el
  rival cuando la cobertura lo permita.
- `late_state_change_edge`: perfil temporal tratado, no raw. Usar eventos de gol
  historicos para medir en recent6 cambios positivos de estado despues del
  minuto 75 menos cambios negativos. Positivo: perder a empatar, empatar a
  ganar, perder a ganar. Negativo: ganar a empatar, empatar a perder, ganar a
  perder.

Resultado inicial 2026-06-15:

- Archivo: `outputs/late_state_change_edge_experiment_2026-06-15.json`
- Cobertura alta: 100.0% del test con algun historial temporal recent6 y 65.4%
  del test con edge no-cero.
- El modelo le dio importancia, pero no mejoro el default: accuracy bajo de
  0.6917 a 0.6842 y log loss subio de 0.7876 a 0.8029 en el split default.
- Mantener fuera del default. Si se retoma, probar una version refinada como una
  sola feature tratada, no un paquete de raw columns.

Version refinada prometedora:

- `late85_points_swing_edge`: recent6 del cambio de puntos de estado despues del
  minuto 85, con ganar/empatar/perder valorado como 3/1/0, calculado como
  team_a menos team_b.
- Archivo: `outputs/refined_late_timeline_second_pass_2026-06-15.json`
- Resultado default split: accuracy se mantuvo en 0.6917 y log loss mejoro de
  0.7876 a 0.7841. Promedio multi-ventana: accuracy +0.0038, log loss
  practicamente plano (+0.0001).
- Individualmente era una mejora pequena. Sigue activa dentro de la receta final
  `parsimonious_guardrail_no_tactical`, pero no como raw timeline data: solo como
  `late85_points_swing_edge`.

## Estilo de pruebas

- Probar una sola feature tratada por vez contra el default limpio.
- No agregar muchas columnas raw en el primer intento; construir una senal
  limpia y futbolera primero.
- Si una feature individual muestra senal, entonces probar combinaciones en una
  segunda fase.

## Regla para activar en default

Activar perfiles solo si cumplen al menos una de estas condiciones contra el
default limpio:

- sube accuracy en el split temporal mas reciente sin empeorar log loss de forma
  relevante;
- o mantiene accuracy y mejora log loss/MAE;
- o mejora promedio multi-ventana y no colapsa en el corte mas reciente.
