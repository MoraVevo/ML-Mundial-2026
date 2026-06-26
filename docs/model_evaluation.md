# Evaluacion del modelo

Este reporte evalua el modelo neutral con cortes temporales disenados para que los partidos de test no entren al entrenamiento.

Identificador del modelo: `neutral_worldcup_v1`

## Politicas de test

### Test Mundial 2026

Los partidos jugados del Mundial 2026 se fuerzan como test. El entrenamiento usa solo partidos de selecciones anteriores al primer partido del Mundial 2026; esos partidos de test no se usan para entrenar.

- Partidos de entrenamiento: 787
- Partidos de test: 60
- Ventana de test: 2026-06-11 a 2026-06-26

### Test externo temporal

Test aleatorio de 104 partidos nacionales no amistosos y fuera del Mundial 2026, tomados del pool reciente de partidos oficiales, seed=42. El entrenamiento usa solo partidos anteriores a la primera fecha seleccionada de test; los partidos seleccionados como test no se usan para entrenar.

- Partidos de entrenamiento: 565
- Partidos de test: 104
- Ventana de test: 2024-09-07 a 2026-03-26

## Metricas

| Evaluacion | Accuracy | Correctos | Log loss | MAE equipo A | MAE equipo B | MAE prom. |
|---|---:|---:|---:|---:|---:|---:|
| Test Mundial 2026 | 0.5667 | 34/60 | 0.9578 | 1.1325 | 0.8921 | 1.0123 |
| Test externo temporal | 0.6538 | 68/104 | 0.8910 | 1.0039 | 1.0573 | 1.0306 |

![Resumen de metricas](assets/model_evaluation/metrics_summary.png)

## Interpretacion tecnica

El modelo es util para direccionar ganadores, pero el umbral actual es conservador con los empates. En estos tests asigna probabilidad al empate para calibracion via log loss, pero la clase con mayor probabilidad casi nunca termina siendo `empate`.

| Evaluacion | Empates reales | Empates predichos como clase principal |
|---|---:|---:|
| Test Mundial 2026 | 16 | 0 |
| Test externo temporal | 22 | 0 |

Por eso se muestra log loss junto a accuracy: accuracy sola oculta si el modelo esta asignando probabilidad util a empates y partidos cerrados. El MAE se reporta aparte porque los regresores de goles pueden estar razonablemente calibrados aunque el clasificador 1X2 elija otra clase.

## Matrices de confusion

### Test Mundial 2026

![Matriz de confusion](assets/model_evaluation/confusion_worldcup_2026.png)

### Test externo temporal

![Matriz de confusion](assets/model_evaluation/confusion_external_random_temporal.png)

## Importancia de features

### Test Mundial 2026

![Importancia de features](assets/model_evaluation/feature_importance_worldcup_2026.png)

| Feature | Importancia |
|---|---:|
| `draw_pressure_index` | 2575.00 |
| `rating_guardrail_edge` | 2185.67 |
| `score_control_value_edge` | 2177.67 |
| `quality_form_edge` | 2143.33 |
| `match_script_compatibility_edge` | 2108.00 |
| `clinical_low_block_matchup_edge` | 2092.67 |
| `rating_drift_abs` | 2045.00 |
| `rating_threat_edge` | 1840.00 |

### Test externo temporal

![Importancia de features](assets/model_evaluation/feature_importance_external_random_temporal.png)

| Feature | Importancia |
|---|---:|
| `draw_pressure_index` | 2026.33 |
| `rating_guardrail_edge` | 1646.67 |
| `rating_drift_abs` | 1556.00 |
| `quality_form_edge` | 1531.67 |
| `match_script_compatibility_edge` | 1526.00 |
| `score_control_value_edge` | 1512.67 |
| `rating_threat_edge` | 1441.00 |
| `clinical_low_block_matchup_edge` | 1439.33 |

## Analisis de error

Las siguientes tablas ordenan los grupos por mayor MAE promedio de goles. Sirven para ver donde el modelo sufre mas, no como ranking definitivo: algunos grupos tienen pocas observaciones.

### Test Mundial 2026

#### Por competicion

| Grupo | Partidos | Accuracy | Log loss | MAE equipo A | MAE equipo B | MAE prom. |
|---|---:|---:|---:|---:|---:|---:|
| FIFA World Cup | 60 | 0.5667 | 0.9578 | 1.1325 | 0.8921 | 1.0123 |

#### Por fase/ronda

| Grupo | Partidos | Accuracy | Log loss | MAE equipo A | MAE equipo B | MAE prom. |
|---|---:|---:|---:|---:|---:|---:|
| GROUP_STAGE | 60 | 0.5667 | 0.9578 | 1.1325 | 0.8921 | 1.0123 |

#### Por resultado real

| Grupo | Partidos | Accuracy | Log loss | MAE equipo A | MAE equipo B | MAE prom. |
|---|---:|---:|---:|---:|---:|---:|
| Equipo A | 31 | 0.7097 | n/a | 1.1896 | 0.8468 | 1.0182 |
| Equipo B | 13 | 0.9231 | n/a | 0.8614 | 1.1659 | 1.0137 |
| Empate | 16 | 0.0000 | n/a | 1.2421 | 0.7573 | 0.9997 |

### Test externo temporal

#### Por competicion

| Grupo | Partidos | Accuracy | Log loss | MAE equipo A | MAE equipo B | MAE prom. |
|---|---:|---:|---:|---:|---:|---:|
| World Cup - Qualification Europe | 40 | 0.8250 | 0.7653 | 1.2179 | 1.0487 | 1.1333 |
| World Cup - Qualification Africa | 19 | 0.7895 | 0.7961 | 0.9963 | 1.2208 | 1.1086 |
| African Nations Championship - Qualification | 2 | 0.5000 | n/a | 0.5774 | 1.3367 | 0.9570 |
| UEFA Nations League | 29 | 0.4138 | 1.0568 | 0.8945 | 0.9727 | 0.9336 |
| CONCACAF Nations League | 9 | 0.6667 | 0.9195 | 0.6254 | 1.1539 | 0.8896 |
| Gulf Cup of Nations | 5 | 0.2000 | 1.1781 | 0.8069 | 0.7099 | 0.7584 |

#### Por fase/ronda

| Grupo | Partidos | Accuracy | Log loss | MAE equipo A | MAE equipo B | MAE prom. |
|---|---:|---:|---:|---:|---:|---:|
| League A - 1 | 1 | 1.0000 | n/a | 2.9460 | 1.2126 | 2.0793 |
| League B - 6 | 1 | 1.0000 | n/a | 2.4275 | 0.8813 | 1.6544 |
| League A - 5 | 2 | 1.0000 | 0.6236 | 2.1993 | 0.9365 | 1.5679 |
| GROUP_STAGE | 61 | 0.7869 | 0.7933 | 1.0947 | 1.0972 | 1.0959 |
| League A - 2 | 2 | 0.5000 | 1.0802 | 0.3925 | 1.7652 | 1.0788 |
| Play-offs A/B | 3 | 0.0000 | 1.4159 | 1.2684 | 0.8335 | 1.0509 |
| QUARTER_FINALS | 9 | 0.3333 | 1.0601 | 0.8091 | 1.2193 | 1.0142 |
| SEMI_FINALS | 4 | 0.5000 | 0.9484 | 1.3674 | 0.6175 | 0.9925 |

#### Por resultado real

| Grupo | Partidos | Accuracy | Log loss | MAE equipo A | MAE equipo B | MAE prom. |
|---|---:|---:|---:|---:|---:|---:|
| Equipo A | 45 | 0.7778 | n/a | 1.3566 | 0.7878 | 1.0722 |
| Empate | 22 | 0.0000 | n/a | 0.7335 | 1.2671 | 1.0003 |
| Equipo B | 37 | 0.8919 | n/a | 0.7357 | 1.2602 | 0.9980 |

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
| `rating_drift_abs` | Magnitud del cambio reciente entre rating historico y rating vivo; captura incertidumbre/volatilidad. |
| `match_script_compatibility_edge` | Compatibilidad tactica estimada entre estilos de partido de ambos equipos. |
| `clinical_low_block_matchup_edge` | Cruce entre definicion ofensiva y capacidad/riesgo contra bloques bajos. |
| `club_attack_talent_edge` | Ventaja de talento ofensivo de plantel/club cuando hay cobertura previa suficiente; faltantes reducen cobertura en vez de inventar valor. |

Grupos conceptuales:

- Fuerza/rating: ranking FIFA, fuerza tipo Elo, guardrails y drift.
- Forma reciente: puntos ajustados por rival y balance de goles.
- Contexto del partido: tipo de competicion, fase/ronda y presion de empate.
- Perfil tactico/ofensivo: compatibilidad de guion de partido y matchup contra bloque bajo.

Quedan fuera de los features: goles objetivo, resultado final, ids crudos, fecha cruda, equipos, fuente y estadisticas postpartido del encuentro evaluado.

## Controles anti-leakage

El test Mundial 2026 es la metrica principal porque evalua el mismo tipo de partido que se quiere predecir. Esos partidos no se usan para entrenar: se separan como test y el modelo se ajusta solo con partidos anteriores al inicio del Mundial 2026.

El test externo temporal revisa si el modelo tambien se sostiene fuera del Mundial. Selecciona partidos oficiales nacionales fuera del Mundial 2026 y entrena solo con partidos anteriores al primer partido seleccionado como test. En otras palabras: el accuracy de cada test se calcula sobre partidos que el modelo no vio durante entrenamiento.

Ambas evaluaciones reconstruyen features antes de entrenar. La fecha se usa para cortes cronologicos y contexto rolling prepartido; no entra como feature directa.
