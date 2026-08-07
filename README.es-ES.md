

# Meta-Research

Una habilidad de Claude Code que te guía a través de un ciclo de investigación dirigido por hipótesis — desde la revisión bibliográfica hasta la publicación — con rigor incorporado, seguimiento de reproducibilidad y mitigación de sesgos.

## Qué hace

Meta-Research actúa como un copiloto de investigación autónomo con un **flujo de trabajo dirigido por hipótesis de 6 fases**:

1. **Revisión Bibliográfica** — Comprender el estado del arte (SOTA), identificar brechas, problemas abiertos y áreas poco exploradas
2. **Generación de Hipótesis** — Generar hipótesis amplias y comprobables, mantener un árbol de hipótesis en YAML
3. **Filtro de Evaluación** — Evaluar cada hipótesis: ¿es novedosa? ¿importante? ¿viable? ¿falsable? ¿ya resuelta?
4. **Diseño de Experimentos** — Protocolo riguroso por hipótesis con planes de análisis bloqueados
5. **Ejecución de Experimentos** — Ejecutar experimentos, rastrear resultados, determinar resultados
6. **Reflexión** — Analizar resultados, decidir: profundizar, ampliar, cambiar de enfoque o concluir

El flujo de trabajo es un **ciclo de investigación** — después de los experimentos, la reflexión decide si continuar iterando (generando nuevas hipótesis, ejecutando más experimentos) o concluir y pasar a la escritura.

Dos artefactos centrales rastrean todo el estado del proyecto:
- **`research-tree.yaml`** — Jerarquía de hipótesis con evaluaciones, experimentos y resultados
- **`research-log.md`** — Línea de tiempo cronológica de exploración y decisiones

## Instalación

### Desde el marketplace

```bash
/plugins marketplace add <marketplace-url>
/plugins install meta-research
```

### Instalación manual

```bash
# Habilidad personal (disponible en todos los proyectos)
ln -s /path/to/meta-research ~/.claude/skills/meta-research

# Habilidad del proyecto (disponible en un solo proyecto)
ln -s /path/to/meta-research /your/project/.claude/skills/meta-research
```

## Uso

```
/meta-research [tu pregunta de investigación o tema]
```

La habilidad siempre comienza con una revisión bibliográfica para comprender el campo antes de generar hipótesis. Si ya tienes artefactos existentes (`research-tree.yaml`, `research-log.md`), se reanudará desde el estado actual.

### Ejemplos

```
/meta-research ¿Cómo escala el aprendizaje en contexto con el tamaño del modelo?
/meta-research Quiero explorar métodos eficientes de ajuste fino para modelos pequeños
/meta-research Ayúdame a analizar los resultados de mis experimentos y decidir los siguientes pasos
```

## Estructura del proyecto

```
meta-research/
├── SKILL.md                              # Definición principal de la habilidad (v2.0 — dirigido por hipótesis)
├── phases/
│   ├── literature-survey.md              # Buscar, filtrar, sintetizar, identificar brechas
│   ├── hypothesis-generation.md          # Generar y organizar hipótesis
│   ├── ideation-frameworks.md            # 12 marcos cognitivos para la generación de ideas
│   ├── judgment.md                       # Evaluar hipótesis antes de invertir recursos
│   ├── experiment-design.md              # Diseño de protocolo por hipótesis
│   ├── experiment-execution.md           # Ejecutar experimentos, analizar, determinar resultados
│   ├── reflection.md                     # Decisiones estratégicas y ciclo de investigación
│   └── writing.md                        # Informe y difusión
├── templates/
│   ├── research-tree.yaml                # Plantilla inicial del árbol de hipótesis
│   ├── judgment-rubric.md                # Rúbrica de puntuación para el filtro de evaluación
│   ├── research-log.md                   # Formato del registro y ejemplos
│   ├── experiment-protocol.md            # Plantilla completa de diseño de experimentos
│   └── reproducibility-checklist.md      # Lista de verificación previa a la publicación
├── raw-meta-research.md                  # Material fuente y referencias
├── LOGBOX.md                             # Registro de desarrollo
├── .claude-plugin/
│   └── plugin.json                       # Manifiesto del plugin
├── LICENSE
└── README.md
```

## Características principales

- **Primero la literatura** — Siempre comienza comprendiendo el estado del arte antes de generar ideas
- **Árbol de hipótesis** — Estructura de datos central en YAML que rastrea todas las hipótesis, evaluaciones y resultados
- **Filtro de evaluación** — Evalúa novedad, importancia, viabilidad y falsabilidad antes de invertir recursos
- **Ciclo de investigación** — Reflexiona después de los experimentos y decide: profundizar, ampliar, cambiar de enfoque o concluir
- **Mitigación de sesgos** — Separa el análisis exploratorio del confirmatorio, limita los grados de libertad del investigador
- **Primero la reproducibilidad** — Control de versiones, entornos fijados y seguimiento de experimentos integrados en el flujo de trabajo
- **Mentalidad de falsificación** — Diseña experimentos para refutar, no para confirmar

## Licencia

[MIT](LICENSE)
