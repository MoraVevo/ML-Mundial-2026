# Evaluacion del modelo

Este reporte evalua el modelo neutral con cortes temporales disenados para que los partidos de test no entren al entrenamiento.

Identificador del modelo: `neutral_worldcup_v9_conservative_depth4_fotmob_xg_probability_ensemble`

## Politicas de test

### Test Mundial 2026

Los partidos jugados del Mundial 2026 se fuerzan como test. El entrenamiento usa solo partidos de selecciones anteriores al primer partido del Mundial 2026; esos partidos de test no se usan para entrenar.

- Partidos de entrenamiento: 787
- Partidos de test: 97
- Ventana de test: 2026-06-11 a 2026-07-09

### Test externo temporal

Test aleatorio de 104 partidos nacionales no amistosos y fuera del Mundial 2026, tomados del pool reciente de partidos oficiales, seed=42. El entrenamiento usa solo partidos anteriores a la primera fecha seleccionada de test; los partidos seleccionados como test no se usan para entrenar.

- Partidos de entrenamiento: 565
- Partidos de test: 104
- Ventana de test: 2024-09-07 a 2026-03-26

### Test combinado objetivo

Diagnostico combinado del objetivo: todos los partidos jugados del Mundial 2026 mas el test externo temporal de partidos oficiales no amistosos. El entrenamiento usa solo partidos anteriores a la primera fecha seleccionada de test y ningun partido de test entra al entrenamiento.

- Partidos de entrenamiento: 565
- Partidos de test: 201
- Ventana de test: 2024-09-07 a 2026-07-09

## Metricas

| Evaluacion | Accuracy | Correctos | Log loss | MAE equipo A | MAE equipo B | MAE prom. |
|---|---:|---:|---:|---:|---:|---:|
| Test Mundial 2026 | 0.6495 | 63/97 | 0.8490 | 0.9559 | 0.8765 | 0.9162 |
| Test externo temporal | 0.6538 | 68/104 | 0.8728 | 0.9992 | 0.9856 | 0.9924 |
| Test combinado objetivo | 0.6368 | 128/201 | 0.8969 | 0.9834 | 0.9428 | 0.9631 |

![Resumen de metricas](assets/model_evaluation/metrics_summary.png)

## Interpretacion tecnica

El modelo es util para direccionar ganadores, pero el umbral actual es conservador con los empates. En estos tests asigna probabilidad al empate para calibracion via log loss, pero la clase con mayor probabilidad casi nunca termina siendo `empate`.

La metrica principal sigue siendo el test Mundial 2026. El test combinado se incluye como diagnostico del objetivo mixto, no como reemplazo de la lectura mundialista.

| Evaluacion | Empates reales | Empates predichos como clase principal |
|---|---:|---:|
| Test Mundial 2026 | 24 | 1 |
| Test externo temporal | 22 | 0 |
| Test combinado objetivo | 46 | 0 |

Por eso se muestra log loss junto a accuracy: accuracy sola oculta si el modelo esta asignando probabilidad util a empates y partidos cerrados. El MAE se reporta aparte porque los regresores de goles pueden estar razonablemente calibrados aunque el clasificador 1X2 elija otra clase.

## Matrices de confusion

### Test Mundial 2026

![Matriz de confusion](assets/model_evaluation/confusion_worldcup_2026.png)

### Test externo temporal

![Matriz de confusion](assets/model_evaluation/confusion_external_random_temporal.png)

### Test combinado objetivo

![Matriz de confusion](assets/model_evaluation/confusion_combined_objective.png)

## Importancia de features

### Test Mundial 2026

![Importancia de features](assets/model_evaluation/feature_importance_worldcup_2026.png)

| Feature | Importancia |
|---|---:|
| `rating_guardrail_edge` | 1008.00 |
| `draw_pressure_index` | 886.33 |
| `rating_threat_edge` | 764.00 |
| `quality_form_edge` | 711.67 |
| `score_timing_edge` | 696.00 |
| `goal_balance_edge` | 541.00 |
| `stage_or_round` | 374.67 |
| `worldcup_fotmob_xg_matchup_team_a` | 186.00 |

### Test externo temporal

![Importancia de features](assets/model_evaluation/feature_importance_external_random_temporal.png)

| Feature | Importancia |
|---|---:|
| `draw_pressure_index` | 956.67 |
| `rating_guardrail_edge` | 855.00 |
| `rating_threat_edge` | 759.00 |
| `score_timing_edge` | 708.67 |
| `quality_form_edge` | 607.67 |
| `goal_balance_edge` | 507.67 |
| `stage_or_round` | 353.67 |
| `worldcup_fotmob_xg_matchup_team_a` | 150.67 |

### Test combinado objetivo

![Importancia de features](assets/model_evaluation/feature_importance_combined_objective.png)

| Feature | Importancia |
|---|---:|
| `draw_pressure_index` | 956.67 |
| `rating_guardrail_edge` | 855.00 |
| `rating_threat_edge` | 759.00 |
| `score_timing_edge` | 708.67 |
| `quality_form_edge` | 607.67 |
| `goal_balance_edge` | 507.67 |
| `stage_or_round` | 353.67 |
| `worldcup_fotmob_xg_matchup_team_a` | 150.67 |

## Analisis de error

Las siguientes tablas ordenan los grupos por mayor MAE promedio de goles. Sirven para ver donde el modelo sufre mas, no como ranking definitivo: algunos grupos tienen pocas observaciones.

### Test Mundial 2026

#### Por competicion

| Grupo | Partidos | Accuracy | Log loss | MAE equipo A | MAE equipo B | MAE prom. |
|---|---:|---:|---:|---:|---:|---:|
| FIFA World Cup | 97 | 0.6495 | 0.8490 | 0.9559 | 0.8765 | 0.9162 |

#### Por fase/ronda

| Grupo | Partidos | Accuracy | Log loss | MAE equipo A | MAE equipo B | MAE prom. |
|---|---:|---:|---:|---:|---:|---:|
| LAST_16 | 8 | 0.6250 | 0.8679 | 0.9083 | 1.5732 | 1.2407 |
| GROUP_STAGE | 72 | 0.6250 | 0.8722 | 1.0606 | 0.8385 | 0.9495 |
| ROUND_OF_32 | 16 | 0.7500 | 0.7380 | 0.5594 | 0.6953 | 0.6273 |
| QUARTER_FINALS | 1 | 1.0000 | n/a | 0.1481 | 0.9388 | 0.5435 |

#### Por resultado real

| Grupo | Partidos | Accuracy | Log loss | MAE equipo A | MAE equipo B | MAE prom. |
|---|---:|---:|---:|---:|---:|---:|
| Empate | 24 | 0.0000 | n/a | 1.0816 | 0.8989 | 0.9902 |
| Equipo B | 26 | 0.8846 | n/a | 0.6688 | 1.1788 | 0.9238 |
| Equipo A | 47 | 0.8511 | n/a | 1.0507 | 0.6978 | 0.8742 |

### Test externo temporal

#### Por competicion

| Grupo | Partidos | Accuracy | Log loss | MAE equipo A | MAE equipo B | MAE prom. |
|---|---:|---:|---:|---:|---:|---:|
| World Cup - Qualification Europe | 40 | 0.8250 | 0.7364 | 1.2008 | 1.0210 | 1.1109 |
| World Cup - Qualification Africa | 19 | 0.7895 | 0.7639 | 0.8237 | 1.0824 | 0.9531 |
| UEFA Nations League | 29 | 0.4138 | 1.0409 | 0.9333 | 0.9200 | 0.9266 |
| African Nations Championship - Qualification | 2 | 0.5000 | n/a | 0.8352 | 0.9899 | 0.9126 |
| CONCACAF Nations League | 9 | 0.6667 | 0.9661 | 0.8081 | 0.9256 | 0.8668 |
| Gulf Cup of Nations | 5 | 0.2000 | 1.1266 | 0.8448 | 0.8214 | 0.8331 |

#### Por fase/ronda

| Grupo | Partidos | Accuracy | Log loss | MAE equipo A | MAE equipo B | MAE prom. |
|---|---:|---:|---:|---:|---:|---:|
| League A - 1 | 1 | 1.0000 | n/a | 3.4009 | 1.4907 | 2.4458 |
| League B - 6 | 1 | 1.0000 | n/a | 2.3374 | 0.7041 | 1.5208 |
| League A - 5 | 2 | 1.0000 | 0.5476 | 2.2599 | 0.4991 | 1.3795 |
| League A - 2 | 2 | 0.5000 | 1.0145 | 0.6649 | 1.4573 | 1.0611 |
| GROUP_STAGE | 61 | 0.7869 | 0.7628 | 1.0283 | 1.0493 | 1.0388 |
| THIRD_PLACE | 1 | 1.0000 | n/a | 1.1938 | 0.8739 | 1.0338 |
| SEMI_FINALS | 4 | 0.5000 | 0.9459 | 1.5878 | 0.4462 | 1.0170 |
| Play-offs A/B | 3 | 0.0000 | 1.2925 | 1.2916 | 0.6215 | 0.9565 |

#### Por resultado real

| Grupo | Partidos | Accuracy | Log loss | MAE equipo A | MAE equipo B | MAE prom. |
|---|---:|---:|---:|---:|---:|---:|
| Equipo A | 45 | 0.7778 | n/a | 1.2521 | 0.7835 | 1.0178 |
| Equipo B | 37 | 0.8919 | n/a | 0.8418 | 1.1397 | 0.9907 |
| Empate | 22 | 0.0000 | n/a | 0.7466 | 1.1399 | 0.9432 |

### Test combinado objetivo

#### Por competicion

| Grupo | Partidos | Accuracy | Log loss | MAE equipo A | MAE equipo B | MAE prom. |
|---|---:|---:|---:|---:|---:|---:|
| World Cup - Qualification Europe | 40 | 0.8250 | 0.7364 | 1.2008 | 1.0210 | 1.1109 |
| World Cup - Qualification Africa | 19 | 0.7895 | 0.7639 | 0.8237 | 1.0824 | 0.9531 |
| FIFA World Cup | 97 | 0.6186 | 0.9227 | 0.9664 | 0.8969 | 0.9317 |
| UEFA Nations League | 29 | 0.4138 | 1.0409 | 0.9333 | 0.9200 | 0.9266 |
| African Nations Championship - Qualification | 2 | 0.5000 | n/a | 0.8352 | 0.9899 | 0.9126 |
| CONCACAF Nations League | 9 | 0.6667 | 0.9661 | 0.8081 | 0.9256 | 0.8668 |
| Gulf Cup of Nations | 5 | 0.2000 | 1.1266 | 0.8448 | 0.8214 | 0.8331 |

#### Por fase/ronda

| Grupo | Partidos | Accuracy | Log loss | MAE equipo A | MAE equipo B | MAE prom. |
|---|---:|---:|---:|---:|---:|---:|
| League A - 1 | 1 | 1.0000 | n/a | 3.4009 | 1.4907 | 2.4458 |
| League B - 6 | 1 | 1.0000 | n/a | 2.3374 | 0.7041 | 1.5208 |
| League A - 5 | 2 | 1.0000 | 0.5476 | 2.2599 | 0.4991 | 1.3795 |
| LAST_16 | 8 | 0.6250 | 0.9117 | 0.8699 | 1.6060 | 1.2380 |
| League A - 2 | 2 | 0.5000 | 1.0145 | 0.6649 | 1.4573 | 1.0611 |
| THIRD_PLACE | 1 | 1.0000 | n/a | 1.1938 | 0.8739 | 1.0338 |
| SEMI_FINALS | 4 | 0.5000 | 0.9459 | 1.5878 | 0.4462 | 1.0170 |
| GROUP_STAGE | 133 | 0.6767 | 0.8631 | 1.0542 | 0.9432 | 0.9987 |

#### Por resultado real

| Grupo | Partidos | Accuracy | Log loss | MAE equipo A | MAE equipo B | MAE prom. |
|---|---:|---:|---:|---:|---:|---:|
| Equipo A | 92 | 0.7935 | n/a | 1.1950 | 0.7489 | 0.9720 |
| Equipo B | 63 | 0.8730 | n/a | 0.7618 | 1.1764 | 0.9691 |
| Empate | 46 | 0.0000 | n/a | 0.8635 | 1.0107 | 0.9371 |

## Construccion de features

El modelo activo usa 10 features prepartido en su base y 12 en el clasificador xG paralelo (mezcla de probabilidades 50%/50%).

| Feature | Significado |
|---|---|
| `competition_family` | Familia de competicion: Mundial, eliminatoria, torneo continental, Nations League u otra categoria nacional. |
| `stage_or_round` | Fase o ronda del partido. Da contexto competitivo: grupo, knockout, jornada, final, etc. |
| `rating_threat_edge` | Ventaja combinada de fuerza/ranking y amenaza ofensiva esperada entre Equipo A y Equipo B. |
| `quality_form_edge` | Forma reciente ajustada por calidad del rival, no solo puntos crudos. |
| `goal_balance_edge` | Diferencia de balance goleador reciente e historico: goles a favor menos goles recibidos. |
| `draw_pressure_index` | Indice de paridad y baja separacion esperada; ayuda a calibrar partidos cerrados. |
| `score_timing_edge` | Ventaja validada de score timing: cuanto controlo el marcador, que tan temprano golpeo, si rescato o perdio puntos tarde, y cuanto tiempo paso persiguiendo el partido. |
| `rating_guardrail_edge` | Correccion de seguridad cuando las senales de amenaza se alejan demasiado del rating base. |
| `club_star_finisher_edge` | Ventaja del mejor finalizador reciente de club dentro del nucleo usado por la seleccion; prioriza techo goleador sobre promedio de talento. |
| `worldcup_fotmob_current_story_edge` | Lectura conservadora del Mundial actual: dominio controlado, presion de ocasiones, soluciones ante bloque bajo, transicion y presion no premiada, siempre con cobertura bilateral. |
| `worldcup_fotmob_xg_matchup_team_a` | xG esperado para el Equipo A: mitad de su xG creado y mitad del xG que concede el rival, con mezcla de historial mundialista y torneo actual. |
| `worldcup_fotmob_xg_matchup_team_b` | xG esperado para el Equipo B con la misma construccion neutral y prepartido. |

Grupos conceptuales:

- Fuerza/rating: ranking FIFA, fuerza tipo Elo y guardrails de ranking.
- Forma reciente: puntos ajustados por rival y balance de goles.
- Contexto del partido: tipo de competicion, fase/ronda y presion de empate.
- Perfil ofensivo del Mundial actual: score timing, control de ocasiones, dominio controlado y finalizador diferencial.
- xG de matchup: xG que cada equipo crea combinado con xG que el rival concede; solo entra al clasificador xG paralelo, no a los regresores de goles.

Quedan fuera de los features: goles objetivo, resultado final, ids crudos, fecha cruda, equipos, fuente y estadisticas postpartido del encuentro evaluado.

## Controles anti-leakage

El test Mundial 2026 es la metrica principal porque evalua el mismo tipo de partido que se quiere predecir. Esos partidos no se usan para entrenar: se separan como test y el modelo se ajusta solo con partidos anteriores al inicio del Mundial 2026.

El test externo temporal revisa si el modelo tambien se sostiene fuera del Mundial. Selecciona partidos oficiales nacionales fuera del Mundial 2026 y entrena solo con partidos anteriores al primer partido seleccionado como test. En otras palabras: el accuracy de cada test se calcula sobre partidos que el modelo no vio durante entrenamiento.

Ambas evaluaciones reconstruyen features antes de entrenar. La fecha se usa para cortes cronologicos y contexto rolling prepartido; no entra como feature directa.
