# Evaluacion del modelo

Este reporte evalua el modelo neutral con cortes temporales disenados para que los partidos de test no entren al entrenamiento.

Identificador del modelo: `neutral_worldcup_v7_conservative_depth4_score_timing_no_clinical_no_script`

## Politicas de test

### Test Mundial 2026

Los partidos jugados del Mundial 2026 se fuerzan como test. El entrenamiento usa solo partidos de selecciones anteriores al primer partido del Mundial 2026; esos partidos de test no se usan para entrenar.

- Partidos de entrenamiento: 787
- Partidos de test: 96
- Ventana de test: 2026-06-11 a 2026-07-07

### Test externo temporal

Test aleatorio de 104 partidos nacionales no amistosos y fuera del Mundial 2026, tomados del pool reciente de partidos oficiales, seed=42. El entrenamiento usa solo partidos anteriores a la primera fecha seleccionada de test; los partidos seleccionados como test no se usan para entrenar.

- Partidos de entrenamiento: 565
- Partidos de test: 104
- Ventana de test: 2024-09-07 a 2026-03-26

### Test combinado objetivo

Diagnostico combinado del objetivo: todos los partidos jugados del Mundial 2026 mas el test externo temporal de partidos oficiales no amistosos. El entrenamiento usa solo partidos anteriores a la primera fecha seleccionada de test y ningun partido de test entra al entrenamiento.

- Partidos de entrenamiento: 565
- Partidos de test: 200
- Ventana de test: 2024-09-07 a 2026-07-07

## Metricas

| Evaluacion | Accuracy | Correctos | Log loss | MAE equipo A | MAE equipo B | MAE prom. |
|---|---:|---:|---:|---:|---:|---:|
| Test Mundial 2026 | 0.6354 | 61/96 | 0.8657 | 0.9710 | 0.8771 | 0.9240 |
| Test externo temporal | 0.6538 | 68/104 | 0.8762 | 0.9825 | 0.9862 | 0.9844 |
| Test combinado objetivo | 0.6400 | 128/200 | 0.9030 | 0.9781 | 0.9482 | 0.9631 |

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
| `rating_guardrail_edge` | 1023.67 |
| `draw_pressure_index` | 933.00 |
| `rating_threat_edge` | 885.00 |
| `quality_form_edge` | 848.33 |
| `score_timing_edge` | 809.67 |
| `goal_balance_edge` | 655.67 |
| `stage_or_round` | 385.00 |
| `competition_family` | 121.00 |

### Test externo temporal

![Importancia de features](assets/model_evaluation/feature_importance_external_random_temporal.png)

| Feature | Importancia |
|---|---:|
| `draw_pressure_index` | 1062.00 |
| `rating_guardrail_edge` | 875.33 |
| `rating_threat_edge` | 788.67 |
| `score_timing_edge` | 772.67 |
| `quality_form_edge` | 700.33 |
| `goal_balance_edge` | 556.00 |
| `stage_or_round` | 371.67 |
| `competition_family` | 120.00 |

### Test combinado objetivo

![Importancia de features](assets/model_evaluation/feature_importance_combined_objective.png)

| Feature | Importancia |
|---|---:|
| `draw_pressure_index` | 1062.00 |
| `rating_guardrail_edge` | 875.33 |
| `rating_threat_edge` | 788.67 |
| `score_timing_edge` | 772.67 |
| `quality_form_edge` | 700.33 |
| `goal_balance_edge` | 556.00 |
| `stage_or_round` | 371.67 |
| `competition_family` | 120.00 |

## Analisis de error

Las siguientes tablas ordenan los grupos por mayor MAE promedio de goles. Sirven para ver donde el modelo sufre mas, no como ranking definitivo: algunos grupos tienen pocas observaciones.

### Test Mundial 2026

#### Por competicion

| Grupo | Partidos | Accuracy | Log loss | MAE equipo A | MAE equipo B | MAE prom. |
|---|---:|---:|---:|---:|---:|---:|
| FIFA World Cup | 96 | 0.6354 | 0.8657 | 0.9710 | 0.8771 | 0.9240 |

#### Por fase/ronda

| Grupo | Partidos | Accuracy | Log loss | MAE equipo A | MAE equipo B | MAE prom. |
|---|---:|---:|---:|---:|---:|---:|
| LAST_16 | 8 | 0.5000 | 0.8914 | 0.9018 | 1.4768 | 1.1893 |
| GROUP_STAGE | 72 | 0.6250 | 0.8891 | 1.0638 | 0.8469 | 0.9554 |
| ROUND_OF_32 | 16 | 0.7500 | 0.7474 | 0.5878 | 0.7130 | 0.6504 |

#### Por resultado real

| Grupo | Partidos | Accuracy | Log loss | MAE equipo A | MAE equipo B | MAE prom. |
|---|---:|---:|---:|---:|---:|---:|
| Empate | 24 | 0.0000 | n/a | 1.0587 | 0.9145 | 0.9866 |
| Equipo B | 26 | 0.8462 | n/a | 0.6975 | 1.1376 | 0.9175 |
| Equipo A | 46 | 0.8478 | n/a | 1.0799 | 0.7103 | 0.8951 |

### Test externo temporal

#### Por competicion

| Grupo | Partidos | Accuracy | Log loss | MAE equipo A | MAE equipo B | MAE prom. |
|---|---:|---:|---:|---:|---:|---:|
| World Cup - Qualification Europe | 40 | 0.8250 | 0.7428 | 1.1910 | 1.0261 | 1.1086 |
| World Cup - Qualification Africa | 19 | 0.7895 | 0.7660 | 0.7922 | 1.0738 | 0.9330 |
| UEFA Nations League | 29 | 0.4138 | 1.0480 | 0.9277 | 0.9049 | 0.9163 |
| CONCACAF Nations League | 9 | 0.6667 | 0.9498 | 0.7387 | 0.9924 | 0.8655 |
| African Nations Championship - Qualification | 2 | 0.5000 | n/a | 0.7759 | 0.9383 | 0.8571 |
| Gulf Cup of Nations | 5 | 0.2000 | 1.1158 | 0.8769 | 0.8148 | 0.8458 |

#### Por fase/ronda

| Grupo | Partidos | Accuracy | Log loss | MAE equipo A | MAE equipo B | MAE prom. |
|---|---:|---:|---:|---:|---:|---:|
| League A - 1 | 1 | 1.0000 | n/a | 3.4470 | 1.3710 | 2.4090 |
| League B - 6 | 1 | 1.0000 | n/a | 2.3088 | 0.5759 | 1.4424 |
| League A - 5 | 2 | 1.0000 | 0.5650 | 2.1834 | 0.3404 | 1.2619 |
| League A - 2 | 2 | 0.5000 | 1.0451 | 0.6999 | 1.7204 | 1.2102 |
| GROUP_STAGE | 61 | 0.7869 | 0.7670 | 1.0190 | 1.0498 | 1.0344 |
| Play-offs A/B | 3 | 0.0000 | 1.3251 | 1.4035 | 0.6428 | 1.0232 |
| QUARTER_FINALS | 9 | 0.3333 | 1.0861 | 0.7472 | 1.1509 | 0.9490 |
| SEMI_FINALS | 4 | 0.5000 | 0.9183 | 1.4457 | 0.4327 | 0.9392 |

#### Por resultado real

| Grupo | Partidos | Accuracy | Log loss | MAE equipo A | MAE equipo B | MAE prom. |
|---|---:|---:|---:|---:|---:|---:|
| Equipo A | 45 | 0.7778 | n/a | 1.2585 | 0.7905 | 1.0245 |
| Equipo B | 37 | 0.8919 | n/a | 0.7955 | 1.1325 | 0.9640 |
| Empate | 22 | 0.0000 | n/a | 0.7324 | 1.1407 | 0.9365 |

### Test combinado objetivo

#### Por competicion

| Grupo | Partidos | Accuracy | Log loss | MAE equipo A | MAE equipo B | MAE prom. |
|---|---:|---:|---:|---:|---:|---:|
| World Cup - Qualification Europe | 40 | 0.8250 | 0.7428 | 1.1910 | 1.0261 | 1.1086 |
| FIFA World Cup | 96 | 0.6250 | 0.9321 | 0.9733 | 0.9069 | 0.9401 |
| World Cup - Qualification Africa | 19 | 0.7895 | 0.7660 | 0.7922 | 1.0738 | 0.9330 |
| UEFA Nations League | 29 | 0.4138 | 1.0480 | 0.9277 | 0.9049 | 0.9163 |
| CONCACAF Nations League | 9 | 0.6667 | 0.9498 | 0.7387 | 0.9924 | 0.8655 |
| African Nations Championship - Qualification | 2 | 0.5000 | n/a | 0.7759 | 0.9383 | 0.8571 |
| Gulf Cup of Nations | 5 | 0.2000 | 1.1158 | 0.8769 | 0.8148 | 0.8458 |

#### Por fase/ronda

| Grupo | Partidos | Accuracy | Log loss | MAE equipo A | MAE equipo B | MAE prom. |
|---|---:|---:|---:|---:|---:|---:|
| League A - 1 | 1 | 1.0000 | n/a | 3.4470 | 1.3710 | 2.4090 |
| League B - 6 | 1 | 1.0000 | n/a | 2.3088 | 0.5759 | 1.4424 |
| League A - 5 | 2 | 1.0000 | 0.5650 | 2.1834 | 0.3404 | 1.2619 |
| League A - 2 | 2 | 0.5000 | 1.0451 | 0.6999 | 1.7204 | 1.2102 |
| LAST_16 | 8 | 0.7500 | 0.9117 | 0.7880 | 1.5857 | 1.1868 |
| Play-offs A/B | 3 | 0.0000 | 1.3251 | 1.4035 | 0.6428 | 1.0232 |
| GROUP_STAGE | 133 | 0.6692 | 0.8695 | 1.0511 | 0.9481 | 0.9996 |
| QUARTER_FINALS | 9 | 0.3333 | 1.0861 | 0.7472 | 1.1509 | 0.9490 |

#### Por resultado real

| Grupo | Partidos | Accuracy | Log loss | MAE equipo A | MAE equipo B | MAE prom. |
|---|---:|---:|---:|---:|---:|---:|
| Equipo A | 91 | 0.7912 | n/a | 1.2215 | 0.7624 | 0.9920 |
| Equipo B | 63 | 0.8889 | n/a | 0.7420 | 1.1741 | 0.9580 |
| Empate | 46 | 0.0000 | n/a | 0.8199 | 1.0062 | 0.9131 |

## Construccion de features

El modelo activo usa 10 features prepartido:

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

Grupos conceptuales:

- Fuerza/rating: ranking FIFA, fuerza tipo Elo y guardrails de ranking.
- Forma reciente: puntos ajustados por rival y balance de goles.
- Contexto del partido: tipo de competicion, fase/ronda y presion de empate.
- Perfil ofensivo del Mundial actual: score timing, control de ocasiones, dominio controlado y finalizador diferencial.

Quedan fuera de los features: goles objetivo, resultado final, ids crudos, fecha cruda, equipos, fuente y estadisticas postpartido del encuentro evaluado.

## Controles anti-leakage

El test Mundial 2026 es la metrica principal porque evalua el mismo tipo de partido que se quiere predecir. Esos partidos no se usan para entrenar: se separan como test y el modelo se ajusta solo con partidos anteriores al inicio del Mundial 2026.

El test externo temporal revisa si el modelo tambien se sostiene fuera del Mundial. Selecciona partidos oficiales nacionales fuera del Mundial 2026 y entrena solo con partidos anteriores al primer partido seleccionado como test. En otras palabras: el accuracy de cada test se calcula sobre partidos que el modelo no vio durante entrenamiento.

Ambas evaluaciones reconstruyen features antes de entrenar. La fecha se usa para cortes cronologicos y contexto rolling prepartido; no entra como feature directa.
