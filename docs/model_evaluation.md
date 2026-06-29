# Evaluacion del modelo

Este reporte evalua el modelo neutral con cortes temporales disenados para que los partidos de test no entren al entrenamiento.

Identificador del modelo: `neutral_worldcup_v1`

## Politicas de test

### Test Mundial 2026

Los partidos jugados del Mundial 2026 se fuerzan como test. El entrenamiento usa solo partidos de selecciones anteriores al primer partido del Mundial 2026; esos partidos de test no se usan para entrenar.

- Partidos de entrenamiento: 787
- Partidos de test: 75
- Ventana de test: 2026-06-11 a 2026-06-29

### Test externo temporal

Test aleatorio de 104 partidos nacionales no amistosos y fuera del Mundial 2026, tomados del pool reciente de partidos oficiales, seed=42. El entrenamiento usa solo partidos anteriores a la primera fecha seleccionada de test; los partidos seleccionados como test no se usan para entrenar.

- Partidos de entrenamiento: 565
- Partidos de test: 104
- Ventana de test: 2024-09-07 a 2026-03-26

### Test combinado objetivo

Diagnostico combinado del objetivo: todos los partidos jugados del Mundial 2026 mas el test externo temporal de partidos oficiales no amistosos. El entrenamiento usa solo partidos anteriores a la primera fecha seleccionada de test y ningun partido de test entra al entrenamiento.

- Partidos de entrenamiento: 565
- Partidos de test: 179
- Ventana de test: 2024-09-07 a 2026-06-29

## Metricas

| Evaluacion | Accuracy | Correctos | Log loss | MAE equipo A | MAE equipo B | MAE prom. |
|---|---:|---:|---:|---:|---:|---:|
| Test Mundial 2026 | 0.6000 | 45/75 | 0.9333 | 1.0843 | 0.8574 | 0.9708 |
| Test externo temporal | 0.6731 | 70/104 | 0.9064 | 0.9963 | 1.0625 | 1.0294 |
| Test combinado objetivo | 0.6369 | 114/179 | 0.9409 | 1.0417 | 0.9659 | 1.0038 |

![Resumen de metricas](assets/model_evaluation/metrics_summary.png)

## Interpretacion tecnica

El modelo es util para direccionar ganadores, pero el umbral actual es conservador con los empates. En estos tests asigna probabilidad al empate para calibracion via log loss, pero la clase con mayor probabilidad casi nunca termina siendo `empate`.

La metrica principal sigue siendo el test Mundial 2026. El test combinado se incluye como diagnostico del objetivo mixto, no como reemplazo de la lectura mundialista.

| Evaluacion | Empates reales | Empates predichos como clase principal |
|---|---:|---:|
| Test Mundial 2026 | 21 | 2 |
| Test externo temporal | 22 | 0 |
| Test combinado objetivo | 43 | 0 |

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
| `draw_pressure_index` | 2494.33 |
| `rating_guardrail_edge` | 2273.33 |
| `score_control_value_edge` | 2242.33 |
| `quality_form_edge` | 2217.67 |
| `clinical_low_block_matchup_edge` | 2159.67 |
| `match_script_compatibility_edge` | 2104.00 |
| `rating_threat_edge` | 2042.00 |
| `goal_balance_edge` | 1720.00 |

### Test externo temporal

![Importancia de features](assets/model_evaluation/feature_importance_external_random_temporal.png)

| Feature | Importancia |
|---|---:|
| `draw_pressure_index` | 2096.33 |
| `rating_guardrail_edge` | 1724.67 |
| `score_control_value_edge` | 1639.67 |
| `match_script_compatibility_edge` | 1567.00 |
| `quality_form_edge` | 1540.00 |
| `rating_threat_edge` | 1524.33 |
| `clinical_low_block_matchup_edge` | 1422.33 |
| `goal_balance_edge` | 1386.67 |

### Test combinado objetivo

![Importancia de features](assets/model_evaluation/feature_importance_combined_objective.png)

| Feature | Importancia |
|---|---:|
| `draw_pressure_index` | 2096.33 |
| `rating_guardrail_edge` | 1724.67 |
| `score_control_value_edge` | 1639.67 |
| `match_script_compatibility_edge` | 1567.00 |
| `quality_form_edge` | 1540.00 |
| `rating_threat_edge` | 1524.33 |
| `clinical_low_block_matchup_edge` | 1422.33 |
| `goal_balance_edge` | 1386.67 |

## Analisis de error

Las siguientes tablas ordenan los grupos por mayor MAE promedio de goles. Sirven para ver donde el modelo sufre mas, no como ranking definitivo: algunos grupos tienen pocas observaciones.

### Test Mundial 2026

#### Por competicion

| Grupo | Partidos | Accuracy | Log loss | MAE equipo A | MAE equipo B | MAE prom. |
|---|---:|---:|---:|---:|---:|---:|
| FIFA World Cup | 75 | 0.6000 | 0.9333 | 1.0843 | 0.8574 | 0.9708 |

#### Por fase/ronda

| Grupo | Partidos | Accuracy | Log loss | MAE equipo A | MAE equipo B | MAE prom. |
|---|---:|---:|---:|---:|---:|---:|
| GROUP_STAGE | 72 | 0.5972 | 0.9355 | 1.1056 | 0.8566 | 0.9811 |
| ROUND_OF_32 | 3 | 0.6667 | 0.8817 | 0.5738 | 0.8760 | 0.7249 |

#### Por resultado real

| Grupo | Partidos | Accuracy | Log loss | MAE equipo A | MAE equipo B | MAE prom. |
|---|---:|---:|---:|---:|---:|---:|
| Empate | 21 | 0.0476 | n/a | 1.4040 | 0.9387 | 1.1713 |
| Equipo A | 35 | 0.7429 | n/a | 1.0694 | 0.7194 | 0.8944 |
| Equipo B | 19 | 0.9474 | n/a | 0.7585 | 1.0216 | 0.8900 |

### Test externo temporal

#### Por competicion

| Grupo | Partidos | Accuracy | Log loss | MAE equipo A | MAE equipo B | MAE prom. |
|---|---:|---:|---:|---:|---:|---:|
| World Cup - Qualification Europe | 40 | 0.8250 | 0.7735 | 1.1626 | 1.0715 | 1.1171 |
| World Cup - Qualification Africa | 19 | 0.7895 | 0.8164 | 0.9604 | 1.2479 | 1.1042 |
| African Nations Championship - Qualification | 2 | 0.5000 | n/a | 0.7462 | 1.1670 | 0.9566 |
| UEFA Nations League | 29 | 0.4828 | 1.0758 | 0.9556 | 0.9574 | 0.9565 |
| Gulf Cup of Nations | 5 | 0.2000 | 1.2245 | 0.9904 | 0.8217 | 0.9060 |
| CONCACAF Nations League | 9 | 0.6667 | 0.9310 | 0.5224 | 1.0799 | 0.8012 |

#### Por fase/ronda

| Grupo | Partidos | Accuracy | Log loss | MAE equipo A | MAE equipo B | MAE prom. |
|---|---:|---:|---:|---:|---:|---:|
| League A - 1 | 1 | 1.0000 | n/a | 2.8990 | 1.5576 | 2.2283 |
| League A - 5 | 2 | 1.0000 | 0.8101 | 2.6005 | 0.8481 | 1.7243 |
| League B - 6 | 1 | 1.0000 | n/a | 2.3639 | 0.7067 | 1.5353 |
| League A - 2 | 2 | 0.5000 | 1.0366 | 0.3974 | 2.0240 | 1.2107 |
| Play-offs A/B | 3 | 0.0000 | 1.4712 | 1.4791 | 0.7527 | 1.1159 |
| GROUP_STAGE | 61 | 0.7869 | 0.8088 | 1.0714 | 1.1402 | 1.1058 |
| QUARTER_FINALS | 9 | 0.4444 | 1.0628 | 0.8664 | 1.1416 | 1.0040 |
| 2nd Round | 2 | 0.5000 | n/a | 0.7462 | 1.1670 | 0.9566 |

#### Por resultado real

| Grupo | Partidos | Accuracy | Log loss | MAE equipo A | MAE equipo B | MAE prom. |
|---|---:|---:|---:|---:|---:|---:|
| Equipo A | 45 | 0.8222 | n/a | 1.3173 | 0.7992 | 1.0582 |
| Empate | 22 | 0.0000 | n/a | 0.7165 | 1.3065 | 1.0115 |
| Equipo B | 37 | 0.8919 | n/a | 0.7721 | 1.2377 | 1.0049 |

### Test combinado objetivo

#### Por competicion

| Grupo | Partidos | Accuracy | Log loss | MAE equipo A | MAE equipo B | MAE prom. |
|---|---:|---:|---:|---:|---:|---:|
| World Cup - Qualification Europe | 40 | 0.8250 | 0.7735 | 1.1626 | 1.0715 | 1.1171 |
| World Cup - Qualification Africa | 19 | 0.7895 | 0.8164 | 0.9604 | 1.2479 | 1.1042 |
| FIFA World Cup | 75 | 0.5867 | 0.9888 | 1.1047 | 0.8319 | 0.9683 |
| African Nations Championship - Qualification | 2 | 0.5000 | n/a | 0.7462 | 1.1670 | 0.9566 |
| UEFA Nations League | 29 | 0.4828 | 1.0758 | 0.9556 | 0.9574 | 0.9565 |
| Gulf Cup of Nations | 5 | 0.2000 | 1.2245 | 0.9904 | 0.8217 | 0.9060 |
| CONCACAF Nations League | 9 | 0.6667 | 0.9310 | 0.5224 | 1.0799 | 0.8012 |

#### Por fase/ronda

| Grupo | Partidos | Accuracy | Log loss | MAE equipo A | MAE equipo B | MAE prom. |
|---|---:|---:|---:|---:|---:|---:|
| League A - 1 | 1 | 1.0000 | n/a | 2.8990 | 1.5576 | 2.2283 |
| League A - 5 | 2 | 1.0000 | 0.8101 | 2.6005 | 0.8481 | 1.7243 |
| League B - 6 | 1 | 1.0000 | n/a | 2.3639 | 0.7067 | 1.5353 |
| League A - 2 | 2 | 0.5000 | 1.0366 | 0.3974 | 2.0240 | 1.2107 |
| Play-offs A/B | 3 | 0.0000 | 1.4712 | 1.4791 | 0.7527 | 1.1159 |
| GROUP_STAGE | 133 | 0.6767 | 0.9063 | 1.0959 | 0.9802 | 1.0380 |
| QUARTER_FINALS | 9 | 0.4444 | 1.0628 | 0.8664 | 1.1416 | 1.0040 |
| 2nd Round | 2 | 0.5000 | n/a | 0.7462 | 1.1670 | 0.9566 |

#### Por resultado real

| Grupo | Partidos | Accuracy | Log loss | MAE equipo A | MAE equipo B | MAE prom. |
|---|---:|---:|---:|---:|---:|---:|
| Equipo A | 80 | 0.7875 | n/a | 1.3299 | 0.7531 | 1.0415 |
| Empate | 43 | 0.0000 | n/a | 0.8829 | 1.0715 | 0.9772 |
| Equipo B | 56 | 0.9107 | n/a | 0.7519 | 1.1887 | 0.9703 |

## Construccion de features

El modelo activo usa 12 features prepartido:

| Feature | Significado |
|---|---|
| `competition_family` | Familia de competicion: Mundial, eliminatoria, torneo continental, Nations League u otra categoria nacional. |
| `stage_or_round` | Fase o ronda del partido. Da contexto competitivo: grupo, knockout, jornada, final, etc. |
| `rating_threat_edge` | Ventaja combinada de fuerza/ranking y amenaza ofensiva esperada entre Equipo A y Equipo B. |
| `quality_form_edge` | Forma reciente ajustada por calidad del rival, no solo puntos crudos. |
| `goal_balance_edge` | Diferencia de balance goleador reciente e historico: goles a favor menos goles recibidos. |
| `draw_pressure_index` | Indice de paridad y baja separacion esperada; ayuda a calibrar partidos cerrados. |
| `score_control_value_edge` | Ventaja en control de marcador: capacidad reciente de sostener o transformar estados de partido. |
| `rating_guardrail_edge` | Correccion de seguridad cuando las senales de amenaza se alejan demasiado del rating base. |
| `match_script_compatibility_edge` | Compatibilidad tactica estimada entre estilos de partido de ambos equipos. |
| `clinical_low_block_matchup_edge` | Cruce entre definicion ofensiva y capacidad/riesgo contra bloques bajos. |
| `club_star_finisher_edge` | Ventaja del mejor finalizador reciente de club dentro del nucleo usado por la seleccion; prioriza techo goleador sobre promedio de talento. |
| `worldcup_points_memory_edge` | Memoria ponderada de puntos en los ultimos partidos mundialistas disponibles antes del partido. |

Grupos conceptuales:

- Fuerza/rating: ranking FIFA, fuerza tipo Elo y guardrails de ranking.
- Forma reciente: puntos ajustados por rival y balance de goles.
- Contexto del partido: tipo de competicion, fase/ronda y presion de empate.
- Perfil tactico/ofensivo: compatibilidad de guion de partido y matchup contra bloque bajo.

Quedan fuera de los features: goles objetivo, resultado final, ids crudos, fecha cruda, equipos, fuente y estadisticas postpartido del encuentro evaluado.

## Controles anti-leakage

El test Mundial 2026 es la metrica principal porque evalua el mismo tipo de partido que se quiere predecir. Esos partidos no se usan para entrenar: se separan como test y el modelo se ajusta solo con partidos anteriores al inicio del Mundial 2026.

El test externo temporal revisa si el modelo tambien se sostiene fuera del Mundial. Selecciona partidos oficiales nacionales fuera del Mundial 2026 y entrena solo con partidos anteriores al primer partido seleccionado como test. En otras palabras: el accuracy de cada test se calcula sobre partidos que el modelo no vio durante entrenamiento.

Ambas evaluaciones reconstruyen features antes de entrenar. La fecha se usa para cortes cronologicos y contexto rolling prepartido; no entra como feature directa.
