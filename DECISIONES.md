# Registro de decisiones — Parches corales

Bitácora de decisiones tomadas sobre el dataset y el pipeline de etiquetado/recorte, para referencia al redactar la tesis (metodología / preprocesamiento de datos).

---

## 2026-09-13 — Verificación de migración de esquema de códigos (v2 → v3)

**Decisión:** se confirmó que la migración de `cobertura_codes_v2.txt` a `cobertura_codes_v3.json` está completa y correcta.

**Cómo se verificó:** se ejecutó la lógica real de `load_coverage_codes()` + `build_composite_label()` (`scripts/batch_crop_and_label_from_excel.py`) contra las 65 hojas CSV en `data/csvs/`, no solo contra la tabla de códigos. Los resultados de categoría mayor coinciden con el dataset final recortado.

**Cambios de nomenclatura de v2 a v3** (todos resueltos vía `aliases` en el JSON, sin pérdida de datos):

| Código v2 | Código v3 | Nota |
|---|---|---|
| `PORSP` | `PRSP` | renombrado |
| `POCSP` | `PCSP` | renombrado |
| `ENGR` (gorgonáceo incrustante) | `GORG` | fusionado con `GORG` (erecto) — v3 ya no distingue erecto/incrustante |
| `OTROS` (subcategoría) | `OTIN` | renombrado (evita choque con el nombre del grupo `OTROS`) |
| `GAPS` (huecos) | `NI` | cambió de grupo: de `SINERTE` a "no identificado" |
| `NA` | `NI` | renombrado |
| `ENFER` | `SICK` | renombrado |
| `BLANQ` | `BLAN` | renombrado |

Códigos nuevos en v3 sin equivalente v2 (adiciones, no migraciones): `NCOR` (coral no identificado), `PEZ` (peces), `DORG` (materia orgánica), `FLUO` (fluorescencia), `COBB`/`PEBB` (ver decisión siguiente).

---

## 2026-09-13 — Categoría `UNKNOWN` (58 parches)

**Qué es:** parches generados cuando, en el Excel/CSV de origen, las columnas `Major Category`, `Subcategory`, `ID Code`, `ID Name` y `Notes` vienen **todas vacías** para ese punto — no hay ninguna anotación que etiquetar.

**No confundir con `NO_IDENTIFICADO` (NI, 1,214 parches):** esa sí es una categoría válida del esquema de cobertura (imagen borrosa, sobre/sub-expuesta, o sustrato genuinamente no identificable en el punto).

**Origen:** función `slugify_label()` / `build_composite_label()` en `scripts/batch_crop_and_label_from_excel.py`, que devuelve `"UNKNOWN"` cuando no hay ningún valor con el que construir una etiqueta.

**Acción:** ninguna por ahora — son filas sin anotación en el origen, no un problema del pipeline de etiquetado. Si se quiere depurar el origen, revisar esas 58 filas en los CSV listados (identificadas por archivo + fila).

---

## 2026-09-13 — Fusión de `BOUL`/`COBB`/`PEBB` en una sola clase `CLASTOS`

**Contexto:** en el esquema v3, `SUSTRATO_INERTE` distingue roca suelta por tamaño:
- `BOUL` (Cantos grandes): >25.6 cm
- `COBB` (Cantos medianos): 6.4–25.6 cm
- `PEBB` (Guijarros): 0.4–6.4 cm

**Decisión:** estas tres se fusionan en una sola clase, `CLASTOS`, al momento de recortar los parches.

**Justificación:**
1. Es la misma escala granulométrica (tipo Wentworth) partida en 3 cortes arbitrarios de diámetro, no tres materiales distintos — conceptualmente es una sola categoría ("roca suelta").
2. Sin una referencia de escala fija en el parche, el tamaño real (cm) no se puede medir de forma confiable a partir de la imagen recortada.
3. Evidencia empírica en los datos: de 57,100+ puntos anotados en `data/csvs/`, **ninguno** usó los códigos `BOUL` ni `COBB` — solo `PEBB` (46 puntos). Esto sugiere que, en la práctica, la distinción de tamaño no se aplicó de forma consistente durante la anotación de campo.
4. Con 0/0/46 muestras, ninguna de las tres clases era entrenable por separado de todas formas.

**Implementación:** `scripts/batch_crop_and_label_from_excel.py`
- Constante `SUBCATEGORY_MERGE = {"BOUL": "CLASTOS", "COBB": "CLASTOS", "PEBB": "CLASTOS"}`.
- Se aplica sobre `sub_code` dentro de `build_composite_label()`, antes de resolver categoría padre y nombre legible.
- `load_coverage_codes()` registra `CLASTOS` como subcategoría de `SINERTE` (`subcategory_by_code` / `parent_by_subcategory`), para que se resuelva igual que cualquier código nativo del esquema.
- `cobertura_codes_v3.json` **no se modifica** — sigue siendo la referencia del esquema de cobertura completo (por si en el futuro se recolectan datos con escala calibrada que sí permitan distinguir tamaño).

**Efecto en el dataset:** solo aplica a **futuros recortes**. El dataset ya generado en `datasets/224x224` (carpeta `GUIJARROS`, 46 parches) no cambia hasta que se vuelva a correr `batch_crop_and_label_from_excel.py` sobre `data/csvs` + `data/imagenes`. Al re-generar, esos 46 parches pasarán a `SUSTRATO_INERTE/CLASTOS`.

---

## Síntesis del estado actual del proyecto (2026-09-13)

Punto de partida a la fecha de esta entrada — todo lo que sigue abajo son decisiones posteriores a este corte.

**Objetivo:** clasificación automática de cobertura bentónica de arrecife coralino (Pacífico colombiano) a partir de parches de imagen recortados de fotos de transectos, para tesis de pregrado.

**Esquema de categorías:** `cobertura_codes_v3.json` — 8 grupos mayores (`ALG`/Algas, `CORAL`, `CORBLAN`/Corales blandos, `SPON`/Esponjas, `OTROS`/Otros organismos, `SINERTE`/Sustrato inerte, `NI`/No identificado, `TAPE`), cada uno con subcategorías (especie o tipo), y para `CORAL` un tercer nivel de condición (sano, blanqueado, recién muerto, fluorescencia). Migración desde el esquema anterior (`cobertura_codes_v2.txt`) verificada completa y correcta (ver entrada de arriba).

**Pipeline de datos:**
1. Origen: fotos de transectos (`data/imagenes/`) + puntos de anotación por foto (`data/csvs/`, 65 archivos, formato CPCe-like: columnas `Major Category`, `Subcategory`, `ID Code`, `ID Name`, `Notes`, `X`, `Y`).
2. Recorte + etiquetado: `scripts/batch_crop_and_label_from_excel.py` — para cada punto anotado, recorta un parche (40×40 px por defecto en origen; el dataset entregado está a 224×224 y también existe una versión a 32×32) y lo guarda en estructura `train/val/test/<categoría>/<subcategoría>[/<condición>]/`.
3. Salida actual: `datasets/224x224/` — **57,100 parches** (`train` 39,900 / `val` 8,600 / `test` 8,600), distribuidos así (antes de la fusión en `CLASTOS`, que aún no se ha vuelto a correr sobre el dataset ya generado):

   | Categoría | Total |
   |---|---|
   | CORAL | 30,266 |
   | ALGAS | 22,981 |
   | SUSTRATO_INERTE | 1,967 |
   | NO_IDENTIFICADO | 1,214 |
   | TAPE | 545 |
   | UNKNOWN | 58 |
   | OTROS_ORGANISMOS | 53 |
   | ESPONJAS | 16 |

   Dentro de `CORAL`, `POCILLOPORA_SPP` domina fuertemente (29,097 de 30,266) — desbalance de clases importante a tener en cuenta para el entrenamiento (ver `compute_class_weights()` en los scripts de entrenamiento). No hay ejemplos de `CORBLAN` (corales blandos) en este dataset.

**Modelado:** varios scripts de entrenamiento en `scripts/` sobre los parches: `train_cnn_patches.py` (CNN simple), `train_mlp_patches.py` (MLP), `train_miniresnet_patches.py` (MiniResNet), `train_finetune.py` (fine-tuning de AlexNet/MobileNetV2/ResNet18/VGG16). Resultados/logs en `results/` y `logs/`.

**Decisiones de limpieza de datos tomadas hasta ahora** (detalle arriba):
- `UNKNOWN` (58 parches) = filas sin ninguna anotación en el CSV de origen; no requiere acción, no es error del pipeline.
- `BOUL`/`COBB`/`PEBB` (tamaños de roca suelta) fusionados en una sola clase `CLASTOS` al momento de recortar, por ser la misma naturaleza de sustrato y no tener forma confiable de medir tamaño en el parche sin escala de referencia. **Pendiente:** volver a correr el recorte para que el dataset en `datasets/224x224` refleje este cambio (`GUIJARROS` → `SUSTRATO_INERTE/CLASTOS`).

---

*(Las decisiones posteriores a este corte se agregan como nuevas entradas debajo de esta síntesis, siguiendo el mismo formato `## YYYY-MM-DD — Título`.)*

---

## 2026-09-13 — Re-ejecución del recorte con la fusión `CLASTOS`

**Qué se hizo:** se corrió de nuevo `scripts/batch_crop_and_label_from_excel.py` con los mismos parámetros documentados en `Readme.MD` (`--patch_size 224 --val_ratio 0.15 --test_ratio 0.15 --seed 42`), para que `datasets/224x224` reflejara la fusión `BOUL`/`COBB`/`PEBB` → `CLASTOS` decidida arriba.

**Respaldo previo:** antes de sobrescribir, se copió el dataset completo (57,100 archivos, 1.048 GB) a `datasets/224x224_backup_pre_clastos_20260913/` (con `robocopy`, ya que el rename directo de la carpeta fue bloqueado por el sistema — posiblemente por sincronización de la carpeta `Documents`).

**Hallazgo durante la verificación:** el script de recorte es **puramente aditivo** — crea/sobrescribe archivos pero nunca borra carpetas de clases que ya no se generan. Al re-correr, la carpeta antigua `SUSTRATO_INERTE/GUIJARROS` (46 archivos) quedó huérfana junto a la nueva `SUSTRATO_INERTE/CLASTOS` (mismos 46 archivos, contenido idéntico). Se eliminó manualmente `GUIJARROS` en `train/val/test` tras confirmar que su contenido era un duplicado exacto de `CLASTOS`.

**Nota para el futuro:** si se vuelve a re-correr el recorte tras cambiar el esquema de etiquetas (nuevas fusiones, renombres, etc.), revisar manualmente si quedan carpetas de clases obsoletas — el script no limpia el directorio de salida antes de escribir.

**Resultado verificado:** 57,100 parches totales, mismos totales por split (train 39,900 / val 8,600 / test 8,600) y por categoría mayor que antes de la fusión; `SUSTRATO_INERTE` pasó de 6 subcategorías a 5 (`CASCAJOS_RUBBLE_ESCOMBROS` 165, `CLASTOS` 46, `MATERIA_ORGANICA` 8, `ROCA` 509, `SEDIMENTO_LIBRE` 1239).

---

## 2026-09-13 — Hallazgo: los modelos actuales solo clasifican por categoría mayor

**Qué se descubrió:** revisando cómo se arma el desbalance de clases, se confirmó que `torchvision.datasets.ImageFolder` (usado en los 4 scripts de entrenamiento) solo considera como "clase" el primer nivel de subcarpetas de `train/` — es decir, **los modelos actuales predicen únicamente la categoría mayor** (8 clases: ALGAS, CORAL, ESPONJAS, NO_IDENTIFICADO, OTROS_ORGANISMOS, SUSTRATO_INERTE, TAPE, UNKNOWN). Todo lo que hay debajo (especie, condición de salud del coral) queda "aplastado" dentro de esa clase mayor; `subcategory_breakdown()` es solo un reporte post-hoc (usa la ruta del archivo para ver, dentro de cada subcategoría real, qué tan seguido el modelo acertó la categoría mayor) — nunca fue una predicción de especie/condición.

**Implicación:** el desbalance especie+condición que se analizó (34 combinaciones, de 1 a 27,541 parches) nunca fue el problema real de los modelos ya entrenados; era información que ni siquiera se usaba como etiqueta.

---

## 2026-09-13 — Aprendizaje jerárquico opcional (especie / condición) + exclusión por umbral

**Decisión:** en vez de reemplazar el modelo de categoría mayor, se agregó soporte **opcional** (apagado por defecto) para dos tareas nuevas, activables con `--task` en los 4 scripts de entrenamiento:

- `--task major` (default): comportamiento de siempre, sin cambios — `ImageFolder` clasificando por categoría mayor.
- `--task species`: clasifica por categoría+subcategoría (especie), ignorando el nivel de condición de salud (un parche sano y uno blanqueado de la misma especie caen en la misma clase).
- `--task condition`: clasifica solo parches de `CORAL` por su condición de salud (sano/blanqueado/recién muerto/fluorescencia), **agrupando todas las especies** — esto es lo que hace entrenable `CORAL_BLANQUEADO` (25 parches en una sola especie -> ~1,546 agrupado).

En `species` y `condition` se aplica exclusión por umbral: `--min_class_count` (default **50**, ver discusión del umbral) saca del dataset las clases con menos de ese total de parches (train+val+test combinados) — no se intenta balancear clases con muy pocos ejemplos, se excluyen (ver DECISIONES.md, entrada de fusión `CLASTOS`, mismo criterio).

**Por qué "on/off" y no reemplazar el modelo actual:** para poder seguir reportando y comparando contra el modelo de categoría mayor ya entrenado, y decidir después (con resultados de ambos) si vale la pena adoptar el enfoque jerárquico de forma permanente.

**Implementación:**
- `scripts/patch_dataset.py` (nuevo): escanea `train/val/test` y resuelve la etiqueta según la tarea directamente de la profundidad de carpetas (`major`/`species`: 1-2 niveles; `condition`: exige 3 niveles y que el primero sea `CORAL`), calcula conteos combinados de los 3 splits para el filtro de umbral, y expone `PatchFolder` (Dataset) + `make_hierarchical_loaders()` + `hierarchical_breakdown()` (equivalente a `subcategory_breakdown()` pero para las tareas nuevas: en `species` desglosa por condición real, en `condition` desglosa por especie real).
- Los 4 scripts (`train_cnn_patches.py`, `train_mlp_patches.py`, `train_finetune.py`, `train_miniresnet_patches.py`) recibieron los mismos cambios: flags `--task {major,species,condition}` y `--min_class_count`, `make_loaders(...)` delega a `make_hierarchical_loaders()` cuando `task != "major"`, `--results_dir` gana un subnivel por tarea (p.ej. `results/results_cnn/species/`), y se guarda `excluded_classes.txt` cuando el umbral excluye alguna clase.
- Verificado (`scan_and_filter` sobre `datasets/224x224`, `min_class_count=50`): `species` retiene 15 clases (excluye 8: `CORAL_NO_IDENTIFICADO`, `PAVONA_VARIANS`, `PORITES_SPP`, `MATERIA_ORGANICA`, `PECES`, `ESPONJAS/INCRUSTANTES`, `OTROS_INVERTEBRADOS`, `CLASTOS`); `condition` retiene 3 clases (`CORAL_SANO` 28,613 / `CORAL_BLANQUEADO` 1,546 / `CORAL_RECIEN_MUERTO` 83) y excluye `FLUORESCENCIA` (12).

**Pendiente:** correr entrenamientos reales con `--task species` y `--task condition` (por ahora solo se validó que los loaders arman las clases correctas) y comparar F1 contra el modelo `major` actual.

---

## 2026-09-13 — Referencia: estados de salud del coral (`--task condition`)

Tabla de los códigos de condición de salud de `cobertura_codes_v3.json` (sección `notas`), con el total de parches anotados en `datasets/224x224` para cada uno — para agregar a la tabla de categorías/subcategorías de la tesis.

| Código | Nombre (ES) | Nombre (EN) | Descripción | Total parches |
|---|---|---|---|---|
| SANO | Coral sano | Healthy coral | Sin signos de enfermedad ni blanqueamiento (incluye corales pálidos que no califican como blanqueados) | 28,613 |
| BLAN (alias BLANQ) | Coral blanqueado | Bleached coral | Apariencia blanca por pérdida de zooxantelas; en corales ramificados, las puntas blancas deben extenderse >2cm para contar como blanqueamiento | 1,546 |
| DCOR | Coral recién muerto | Recently dead coral | Sin algas ni pólipos | 83 |
| FLUO | Fluorescencia | Fluorescent bleaching | Pigmentación rosada por actividad de proteína fluorescente | 12 |
| SICK (alias ENFER) | Coral enfermo | Sick coral | Síntomas de enfermedad (p.ej. banda negra o banda blanca) | 0 (sin parches anotados) |
| OTRO | Otro | Other | Cualquier rasgo no cubierto por las categorías anteriores | 0 (sin parches anotados) |

**Nota:** `SICK`/`OTRO` están definidos en el esquema pero no aparecen anotados en ningún parche del dataset actual — es una limitación de los datos de origen, no un error de categorización. `FLUO` queda excluido de `--task condition` por defecto (`min_class_count=50`).

---

## 2026-09-13 — Primer entrenamiento real: MobileNetV2 fine-tuned, `species` y `condition`

**Qué se corrió:** `train_finetune.py --model mobilenet_v2` (el más liviano de los 4 disponibles: resnet18/vgg16/alexnet/mobilenet_v2), con `--task species` y luego `--task condition`, `--min_class_count 50`, parámetros por defecto (15 épocas, `freeze_level=2`, `patience=10`). Corrido en GPU vía DirectML (`.venv-directml`, AMD Radeon RX 6600) — el `.venv` normal no tiene `torch_directml` instalado y hubiera corrido en CPU.

**Resultado `species`** (15 clases, accuracy test 60.5%, F1 macro 23.8%):

| Categoría mayor | n subcategorías | accuracy promedio (macro) | accuracy ponderada | n parches test |
|---|---|---|---|---|
| UNKNOWN | 1 | 0.0% | 0.0% | 8 |
| ALGAS | 4 | 30.8% | 43.2% | 3346 |
| CORAL | 5 | 36.7% | 73.1% | 4780 |
| NO_IDENTIFICADO | 1 | 39.0% | 39.0% | 187 |
| SUSTRATO_INERTE | 3 | 41.4% | 59.3% | 189 |
| TAPE | 1 | 93.2% | 93.2% | 74 |

Solo `POCILLOPORA_SPP` (F1 83.0%) y `TAPE` (F1 60.0%) salen bien. 4 clases sacaron 0% (recall y precision): `UNKNOWN`, `ALGAS/FRONDOSAS_MACROALGAS_1_CM`, `CORAL/PAVONA_GIGANTEA`, `CORAL/POCILLOPORA_GRANDIS` — sobrevivieron el umbral de 50 pero con 10 épocas de fine-tuning parcial no alcanzaron a aprenderse. Conclusión: el umbral de exclusión evita clases inviables, pero **no resuelve el desbalance entre las clases que sobreviven** (`POCILLOPORA_SPP` sigue dominando CORAL con 4,546 de 4,780 en test).

**Resultado `condition`** (3 clases, accuracy test 78.5%, F1 macro 41.3%, F1 ponderado 83.8%):

| Clase | Precision | Recall | F1 | Soporte |
|---|---|---|---|---|
| CORAL_SANO | 97.5% | 79.3% | 87.4% | 4495 |
| CORAL_BLANQUEADO | 16.8% | 69.6% | 27.1% | 270 |
| CORAL_RECIEN_MUERTO | 20.0% | 6.3% | 9.5% | 16 |

`condition` le va notablemente mejor que `species` (F1 macro 41.3% vs 23.8%) — esperable, son 3 clases con más ejemplos cada una tras agrupar por especie (ver decisión de agrupar condición a través de especies). `CORAL_BLANQUEADO` tiene recall alto pero precisión baja (sobre-predice blanqueado; 929 `CORAL_SANO` reales confundidos con blanqueado). `CORAL_RECIEN_MUERTO` sigue siendo el punto débil (solo 83 parches en total).

**Pendiente / próximos pasos:** correr `--task major` (baseline) con el mismo modelo para tener punto de comparación limpio; considerar más épocas o `freeze_level` distinto para las clases que sacaron 0%; evaluar si vale la pena juntar `PAVONA_GIGANTEA`/`POCILLOPORA_GRANDIS`/etc. en un bucket "otras especies de coral" en vez de excluirlas, ahora que se ve que ni con el umbral de 50 aprenden bien.
