# Graph Report - Parches_corales  (2026-09-11)

## Corpus Check
- 10 files · ~16,178 words
- Verdict: corpus is large enough that graph structure adds value.

## Summary
- 223 nodes · 307 edges · 17 communities (14 shown, 3 thin omitted)
- Extraction: 93% EXTRACTED · 7% INFERRED · 0% AMBIGUOUS · INFERRED: 20 edges (avg confidence: 0.78)
- Token cost: 0 input · 0 output

## Graph Freshness
- Built from commit: `0ec4fc9a`
- Run `git rev-parse HEAD` and compare to check if the graph is stale.
- Run `graphify update .` after code changes (no API cost).

## Community Hubs (Navigation)
- NA
- train_miniresnet_patches.py
- scripts/batch_crop_and_label_from_excel.py command (patches + labels dataset build)
- batch_crop_and_label_from_excel.py
- main
- CORAL - Coral
- main
- SINERTE - Sustrato inerte (Inert substrate)
- main
- CORBLAN - Corales blandos (Soft corals)
- ALG - Algas (Algae)
- CLAUDE.md
- SPON - Esponjas (Sponges)
- OTROS - Otros organismos (Other organisms)
- TAPE - Tape

## God Nodes (most connected - your core abstractions)
1. `main()` - 12 edges
2. `main()` - 11 edges
3. `main()` - 11 edges
4. `main()` - 11 edges
5. `scripts/batch_crop_and_label_from_excel.py command (patches + labels dataset build)` - 11 edges
6. `CORAL - Coral` - 11 edges
7. `clean_cell()` - 9 edges
8. `main()` - 9 edges
9. `requirements.txt (Python dependency manifest)` - 9 edges
10. `slugify_label()` - 8 edges

## Surprising Connections (you probably didn't know these)
- `openpyxl` --conceptually_related_to--> `scripts/batch_crop_and_label_from_excel.py command (patches + labels dataset build)`  [INFERRED]
  requirements.txt → Readme.MD
- `pandas` --conceptually_related_to--> `scripts/batch_crop_and_label_from_excel.py command (patches + labels dataset build)`  [INFERRED]
  requirements.txt → Readme.MD
- `pillow` --conceptually_related_to--> `scripts/batch_crop_and_label_from_excel.py command (patches + labels dataset build)`  [INFERRED]
  requirements.txt → Readme.MD
- `torchvision` --conceptually_related_to--> `scripts/train_finetune.py --model resnet18 command`  [INFERRED]
  requirements.txt → Readme.MD
- `torchvision` --conceptually_related_to--> `scripts/train_finetune.py --model vgg16 command`  [INFERRED]
  requirements.txt → Readme.MD

## Import Cycles
- None detected.

## Hyperedges (group relationships)
- **Fine-tuning variants sharing scripts/train_finetune.py** — readme_train_finetune_resnet18, readme_train_finetune_vgg16, readme_train_finetune_alexnet, readme_train_finetune_mobilenet_v2 [EXTRACTED 1.00]
- **Coral patch dataset build feeding all training pipelines** — readme_batch_crop_and_label_from_excel, readme_train_mlp_patches, readme_train_cnn_patches, readme_train_finetune_resnet18, readme_train_miniresnet_patches [INFERRED 0.85]

## Communities (17 total, 3 thin omitted)

### Community 0 - "NA"
Cohesion: 0.33
Nodes (6): BLANQ - Coral blanqueado, DCOR - Coral recien muerto, ENFER - Coral enfermo, NA, OTRO - Otro (note code), SANO - Coral sano

### Community 1 - "train_miniresnet_patches.py"
Cohesion: 0.08
Nodes (28): BasicBlock, _ClassLockedImageFolder, compute_class_weights(), eval_epoch(), get_device(), main(), make_loaders(), MiniResNet (+20 more)

### Community 2 - "scripts/batch_crop_and_label_from_excel.py command (patches + labels dataset build)"
Cohesion: 0.16
Nodes (19): scripts/batch_crop_and_label_from_excel.py command (patches + labels dataset build), batch_crop_from_excel_per_image.py command (crop patches from excel), Install Dependencies (pip install requirements.txt), scripts/train_cnn_patches.py command (Simple CNN training), scripts/train_finetune.py --model alexnet command, scripts/train_finetune.py --model mobilenet_v2 command, scripts/train_finetune.py --model resnet18 command, scripts/train_finetune.py --model vgg16 command (+11 more)

### Community 3 - "batch_crop_and_label_from_excel.py"
Cohesion: 0.11
Nodes (37): Path, build_composite_label(), clamp(), clean_cell(), code_from_major_value(), crop_centered_patch(), _empty_codes(), ensure_dir() (+29 more)

### Community 4 - "main"
Cohesion: 0.10
Nodes (23): _ClassLockedImageFolder, compute_class_weights(), ConvNet, eval_epoch(), get_device(), main(), make_loaders(), predict_all() (+15 more)

### Community 5 - "CORAL - Coral"
Cohesion: 0.17
Nodes (12): CORAL - Coral, GPLA - Gardineroseris planulata, PCHI - Pavona chiriquiensis, PCLA - Pavona clavus, PFRO - Pavona frondifera, PGIG - Pavona gigantea, PGRA - Pocillopora grandis, PMAL - Pavona maldivensis (+4 more)

### Community 6 - "main"
Cohesion: 0.09
Nodes (25): _ClassLockedImageFolder, compute_class_weights(), eval_epoch(), get_device(), main(), make_loaders(), MLP, parse_hidden() (+17 more)

### Community 7 - "SINERTE - Sustrato inerte (Inert substrate)"
Cohesion: 0.33
Nodes (6): BOUL - Cantos, GAPS - Huecos, ROCK - Roca, RUBB - Cascajos (rubble), escombros, SAND - Sedimento libre, SINERTE - Sustrato inerte (Inert substrate)

### Community 9 - "main"
Cohesion: 0.12
Nodes (21): _ClassLockedImageFolder, compute_class_weights(), eval_epoch(), get_device(), load_pretrained_model(), main(), make_loaders(), predict_all() (+13 more)

### Community 10 - "CORBLAN - Corales blandos (Soft corals)"
Cohesion: 0.33
Nodes (6): ANEM - Anemonas, CMOR - Coralimorfarios, CORBLAN - Corales blandos (Soft corals), ENGR - Gorgonaceos incrustantes, GORG - Gorgonaceos erectos, ZOAN - Zoantidios

### Community 11 - "ALG - Algas (Algae)"
Cohesion: 0.40
Nodes (5): ALG - Algas (Algae), CALG - Calcareas erectas, EALG - Calcareas incrustantes, FALG - Frondosas, macroalgas (>1 cm), TALG - Tapetes (frondosa o filamentosa, <1cm)

### Community 13 - "SPON - Esponjas (Sponges)"
Cohesion: 0.67
Nodes (3): ENSP - Incrustantes (sponge subtype), ERSP - Erectas (sponge subtype), SPON - Esponjas (Sponges)

## Knowledge Gaps
- **40 isolated node(s):** `graphify`, `Install Dependencies (pip install requirements.txt)`, `batch_crop_from_excel_per_image.py command (crop patches from excel)`, `scikit-learn`, `plotly` (+35 more)
  These have ≤1 connection - possible missing edges or undocumented components.
- **3 thin communities (<3 nodes) omitted from report** — run `graphify query` to explore isolated nodes.

## Suggested Questions
_Questions this graph is uniquely positioned to answer:_

- **Why does `main()` connect `train_miniresnet_patches.py` to `batch_crop_and_label_from_excel.py`?**
  _High betweenness centrality (0.144) - this node is a cross-community bridge._
- **Why does `main()` connect `main` to `batch_crop_and_label_from_excel.py`?**
  _High betweenness centrality (0.135) - this node is a cross-community bridge._
- **Why does `main()` connect `main` to `batch_crop_and_label_from_excel.py`?**
  _High betweenness centrality (0.126) - this node is a cross-community bridge._
- **Are the 8 inferred relationships involving `Path` (e.g. with `main()` and `subcategory_breakdown()`) actually correct?**
  _`Path` has 8 INFERRED edges - model-reasoned connections that need verification._
- **What connects `graphify`, `Install Dependencies (pip install requirements.txt)`, `batch_crop_from_excel_per_image.py command (crop patches from excel)` to the rest of the system?**
  _40 weakly-connected nodes found - possible documentation gaps or missing edges._
- **Should `train_miniresnet_patches.py` be split into smaller, more focused modules?**
  _Cohesion score 0.07777777777777778 - nodes in this community are weakly interconnected._
- **Should `batch_crop_and_label_from_excel.py` be split into smaller, more focused modules?**
  _Cohesion score 0.10526315789473684 - nodes in this community are weakly interconnected._