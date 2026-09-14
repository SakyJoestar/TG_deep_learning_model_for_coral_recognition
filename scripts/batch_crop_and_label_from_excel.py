#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""batch_crop_and_label_from_excel.py.

Para cada imagen en ``--images_dir`` busca su Excel/CSV homónimo en
``--excels_dir``, lee las coordenadas X/Y y una columna de etiqueta
(``--label_col``), recorta parches (40x40 por defecto) centrados en cada
punto y los guarda en una estructura tipo ``ImageFolder``
(``train/val/test/<clase>/...``), lista para entrenar los modelos de
``scripts/train_*.py``.

Si no se indica ``--label_col``, la etiqueta se construye de forma
compuesta (categoría mayor + subcategoría + estado de salud del coral)
usando el archivo de códigos de cobertura (``--codes_file``, por defecto
``cobertura_codes_v3.json``; también acepta el formato ``.txt`` tipo CSV
de ``cobertura_codes_v2.txt`` para compatibilidad hacia atrás).

Requisitos:
    pip install pandas pillow openpyxl

Uso típico:
    python scripts/batch_crop_and_label_from_excel.py \
        --images_dir data/imagenes --excels_dir data/csvs \
        --out_dir ./datasets --patch_size 32
"""

import os, argparse, csv, json, random, re, unicodedata
from pathlib import Path
import pandas as pd
from PIL import Image

IMG_EXTS = (".jpg",".jpeg",".png",".tif",".tiff",".bmp")
XLS_EXTS = (".xlsx",".xls")
CSV_EXTS = (".csv",)
DEFAULT_CORAL_STATES = {"DCOR", "OTRO", "ENFER", "BLANQ", "SANO", "FLUO"}
# BOUL/COBB/PEBB (cantos grandes/medianos/guijarros) son la misma naturaleza de
# sustrato (roca suelta / clastos) diferenciada solo por un umbral de tamaño
# que no se puede medir de forma confiable en un parche sin escala de
# referencia, así que se fusionan en una sola clase ("CLASTOS") al momento de
# recortar. cobertura_codes_v3.json conserva las 3 clases por separado como
# referencia del esquema de cobertura original.
SUBCATEGORY_MERGE = {"BOUL": "CLASTOS", "COBB": "CLASTOS", "PEBB": "CLASTOS"}

def clean_cell(v):
    """Convierte una celda de DataFrame a texto limpio.

    Args:
        v: Valor de la celda (puede ser NaN, número, texto, etc.).

    Returns:
        str: El valor convertido a ``str`` y sin espacios al inicio/fin,
        o ``""`` si el valor era NaN.
    """
    if pd.isna(v):
        return ""
    return str(v).strip()

def normalize_lookup_key(v):
    """Normaliza un texto para usarlo como clave de búsqueda insensible a acentos y mayúsculas.

    Quita tildes/diacríticos, colapsa espacios múltiples y pasa a minúsculas,
    de modo que "Cobertura Coral" y "cobertura   coral" produzcan la misma clave.

    Args:
        v: Valor a normalizar (se limpia primero con :func:`clean_cell`).

    Returns:
        str: Clave normalizada en minúsculas, sin acentos ni espacios extra.
    """
    s = clean_cell(v)
    s = unicodedata.normalize("NFKD", s)
    s = "".join(ch for ch in s if not unicodedata.combining(ch))
    return re.sub(r"\s+", " ", s).strip().lower()

def slugify_label(v):
    """Convierte un texto en un código de etiqueta seguro para nombres de carpeta.

    Elimina acentos, sustituye cualquier caracter no alfanumérico por ``_``
    y devuelve el resultado en mayúsculas (p.ej. "Coral, sano" -> "CORAL_SANO").

    Args:
        v: Valor a convertir en slug.

    Returns:
        str: Código en mayúsculas compuesto solo por ``[A-Z0-9_]``, o
        ``"UNKNOWN"`` si el valor está vacío o queda vacío tras limpiarlo.
    """
    s = clean_cell(v)
    if not s:
        return "UNKNOWN"
    s = unicodedata.normalize("NFKD", s).encode("ascii", "ignore").decode("ascii")
    s = re.sub(r"[^A-Za-z0-9]+", "_", s).strip("_").upper()
    return s or "UNKNOWN"

def is_hex_color(v):
    """Indica si un valor es un código de color hexadecimal de 6 dígitos.

    Se usa para detectar, dentro del archivo de códigos de cobertura, las
    filas de "categoría mayor" (que llevan un color asociado) frente a las
    de subcategoría o nota.

    Args:
        v: Valor a comprobar.

    Returns:
        bool: ``True`` si el valor limpio coincide con ``[0-9A-Fa-f]{6}``.
    """
    return bool(re.fullmatch(r"[0-9A-Fa-f]{6}", clean_cell(v)))

def _empty_codes():
    """Estructura vacía de códigos (con los estados de coral por defecto)."""
    return {
        "major_by_code": {},
        "major_code_by_name": {},
        "subcategory_by_code": {},
        "parent_by_subcategory": {},
        "state_by_code": {},
        "state_codes": set(DEFAULT_CORAL_STATES),
    }

def load_coverage_codes(path, debug=False):
    """Carga el diccionario de códigos de cobertura bentónica.

    Acepta dos formatos, detectados por la extensión de ``path``:

    - ``.json`` (p.ej. ``cobertura_codes_v3.json``, formato actual): ver
      :func:`_load_coverage_codes_json`.
    - Cualquier otra extensión (p.ej. ``cobertura_codes_v2.txt``, formato
      legado tipo CSV): ver :func:`_load_coverage_codes_csv`.

    Args:
        path: Ruta al archivo de códigos. Si es falsy o no existe, se
            devuelve la estructura vacía (con los estados por defecto).
        debug: Si es ``True``, imprime un resumen de cuántos códigos se
            cargaron de cada tipo.

    Returns:
        dict: Diccionario con las claves:

        - ``major_by_code``: código -> nombre de categoría mayor.
        - ``major_code_by_name``: nombre/código normalizado -> código de
          categoría mayor (para búsquedas flexibles).
        - ``subcategory_by_code``: código -> nombre de subcategoría.
        - ``parent_by_subcategory``: código de subcategoría -> código de su
          categoría mayor.
        - ``state_by_code``: código de estado -> nombre del estado.
        - ``state_codes``: conjunto de códigos de estado del coral.
    """
    if not path:
        return _empty_codes()

    path = Path(path)
    if not path.exists():
        if debug:
            print(f"[AVISO] No encontré archivo de códigos: {path}")
        return _empty_codes()

    if path.suffix.lower() == ".json":
        codes = _load_coverage_codes_json(path)
    else:
        codes = _load_coverage_codes_csv(path)

    codes["subcategory_by_code"].setdefault("CLASTOS", "Clastos")
    codes["parent_by_subcategory"].setdefault("CLASTOS", "SINERTE")

    if debug:
        print(
            "[DEBUG] códigos cargados: "
            f"mayores={len(codes['major_by_code'])}, "
            f"subcategorías={len(codes['subcategory_by_code'])}, "
            f"estados={sorted(codes['state_codes'])}"
        )
    return codes

def _load_coverage_codes_csv(path):
    """Parsea el formato legado tipo CSV (v2, p.ej. ``cobertura_codes_v2.txt``).

    El archivo tiene tres columnas por fila: ``codigo, nombre,
    tercera_columna``. La tercera columna decide el tipo de fila:

    - Si es un color hexadecimal -> es una "categoría mayor" (p.ej. CORAL, ALG).
    - Si no lo es -> es una "subcategoría", y la tercera columna indica su
      categoría mayor "padre".
    - Tras una fila cuyo código es ``NOTES``, las filas siguientes se
      interpretan como "estados" del coral (p.ej. sano, enfermo, blanqueado).

    Args:
        path: Ruta al archivo ``.txt``/``.csv`` de códigos (ya verificado
            que existe).

    Returns:
        dict: Mismo formato que :func:`load_coverage_codes`.
    """
    codes = _empty_codes()
    in_notes = False
    with path.open("r", encoding="utf-8-sig", errors="replace", newline="") as fh:
        for row in csv.reader(fh, skipinitialspace=True):
            row = [clean_cell(c).strip('"') for c in row if clean_cell(c)]
            if not row:
                continue

            code = slugify_label(row[0])
            if code == "NOTES":
                in_notes = True
                continue

            if len(row) < 2:
                continue

            name = clean_cell(row[1])
            third = clean_cell(row[2]) if len(row) >= 3 else ""

            if in_notes:
                codes["state_by_code"][code] = name
                codes["state_codes"].add(code)
            elif len(row) >= 3 and is_hex_color(third):
                codes["major_by_code"][code] = name
                codes["major_code_by_name"][normalize_lookup_key(name)] = code
            elif len(row) >= 3:
                parent = slugify_label(third)
                codes["subcategory_by_code"][code] = name
                codes["parent_by_subcategory"][code] = parent

    for code, name in codes["major_by_code"].items():
        codes["major_code_by_name"][normalize_lookup_key(code)] = code
        codes["major_code_by_name"][normalize_lookup_key(name)] = code

    return codes

def _load_coverage_codes_json(path):
    """Parsea el formato actual (v3, p.ej. ``cobertura_codes_v3.json``).

    Estructura esperada (ver ``cobertura_codes_v3.json``)::

        {
          "grupos": [
            {"grupo": "CORAL", "nombre_es": "Coral", "color_hex": "...",
             "codigos": [
               {"code": "PGRA", "name_es": "Pocillopora grandis",
                "aliases": ["PGRA_VIEJO", ...]},
               ...
             ]},
            ...
          ],
          "notas": [
            {"code": "SANO", "name_es": "Coral sano", "aliases": [...]},
            ...
          ]
        }

    Cada código (de grupo o de nota) puede declarar ``aliases``: códigos
    usados en versiones anteriores del esquema que ahora se consideran el
    mismo código, para que datos ya etiquetados con el código viejo sigan
    resolviendo a la categoría correcta sin tener que re-etiquetar nada.

    Un ``codigo`` cuyo ``code`` coincide con el de su propio grupo (p.ej.
    ``NI``/``NI`` o ``TAPE``/``TAPE`` - grupos sin subcategorías reales)
    también registra sus alias como alias de la categoría mayor, para que
    resuelvan igual si aparecen en la columna "Major Category" o en
    "Subcategory" (p.ej. el "NA" viejo, que en v2 era a la vez categoría
    mayor y subcategoría de sí misma).

    Args:
        path: Ruta al archivo ``.json`` de códigos (ya verificado que existe).

    Returns:
        dict: Mismo formato que :func:`load_coverage_codes`.
    """
    codes = _empty_codes()
    data = json.loads(path.read_text(encoding="utf-8-sig"))

    for grupo in data.get("grupos", []):
        grupo_code = slugify_label(grupo["grupo"])
        grupo_name = clean_cell(grupo.get("nombre_es", grupo["grupo"]))
        codes["major_by_code"][grupo_code] = grupo_name
        codes["major_code_by_name"][normalize_lookup_key(grupo_code)] = grupo_code
        codes["major_code_by_name"][normalize_lookup_key(grupo_name)] = grupo_code

        for entry in grupo.get("codigos", []):
            sub_code = slugify_label(entry["code"])
            sub_name = clean_cell(entry.get("name_es", entry["code"]))
            codes["subcategory_by_code"][sub_code] = sub_name
            codes["parent_by_subcategory"][sub_code] = grupo_code

            for alias in entry.get("aliases", []):
                alias_code = slugify_label(alias)
                codes["subcategory_by_code"].setdefault(alias_code, sub_name)
                codes["parent_by_subcategory"].setdefault(alias_code, grupo_code)
                if sub_code == grupo_code:
                    codes["major_code_by_name"].setdefault(
                        normalize_lookup_key(alias_code), grupo_code
                    )

    for nota in data.get("notas", []):
        state_code = slugify_label(nota["code"])
        state_name = clean_cell(nota.get("name_es", nota["code"]))
        codes["state_by_code"][state_code] = state_name
        codes["state_codes"].add(state_code)
        for alias in nota.get("aliases", []):
            alias_code = slugify_label(alias)
            codes["state_by_code"].setdefault(alias_code, state_name)
            codes["state_codes"].add(alias_code)

    return codes

def find_col(cols, candidates):
    """Busca en una lista de columnas la primera que contenga alguno de los nombres candidatos.

    La comparación es insensible a mayúsculas y por subcadena (p.ej. el
    candidato ``"x"`` coincide con una columna llamada ``"X (pixel)"``).

    Args:
        cols: Iterable con los nombres de columna disponibles (p.ej.
            ``df.columns``).
        candidates: Lista de nombres candidatos, ordenados por prioridad.

    Returns:
        El nombre de columna original (tal como aparece en ``cols``) que
        coincide con el primer candidato encontrado, o ``None`` si ninguno
        coincide.
    """
    cl = {str(c).lower(): c for c in cols}
    for cand in candidates:
        cand = cand.lower()
        for c in cols:
            if cand in str(c).lower():
                return cl[str(c).lower()]
    return None

def normalize_numeric_series(s):
    """Convierte una serie de pandas a valores numéricos, tolerando coma decimal.

    Args:
        s (pandas.Series): Serie con valores de texto o numéricos.

    Returns:
        pandas.Series: Serie numérica (``float``); los valores que no se
        pudieron convertir quedan como ``NaN``.
    """
    s = s.astype(str).str.strip().str.replace(",", ".", regex=False)
    return pd.to_numeric(s, errors="coerce")

def code_from_major_value(value, codes):
    """Resuelve el código de categoría mayor a partir de un valor de texto libre.

    Primero intenta convertir el valor directamente a un código válido
    (slugify); si no coincide con ninguna categoría mayor conocida, busca
    por nombre normalizado en ``codes["major_code_by_name"]``.

    Args:
        value: Valor de la columna "Major Category" tal como aparece en el
            Excel/CSV.
        codes: Diccionario de códigos devuelto por :func:`load_coverage_codes`.

    Returns:
        str: Código de categoría mayor (p.ej. ``"CORAL"``), o ``""`` si
        ``value`` está vacío.
    """
    value = clean_cell(value)
    if not value:
        return ""
    value_code = slugify_label(value)
    if value_code in codes["major_by_code"]:
        return value_code
    return codes["major_code_by_name"].get(normalize_lookup_key(value), value_code)

def build_composite_label(row, codes, label_mode="full", sep="/"):
    """Construye la etiqueta de un punto a partir de las columnas Major/Subcategory/Notes.

    Combina la categoría mayor, la subcategoría y (si aplica) el estado de
    salud del coral en una sola etiqueta, según ``label_mode``. Si las
    "Notes" contienen un código de estado del coral (p.ej. "sano",
    "enfermo") y la subcategoría pertenece a CORAL (o no tiene categoría
    mayor propia), la categoría mayor se fuerza a ``"CORAL"``.

    Las partes se resuelven a su nombre legible (p.ej. ``"PGRA"`` ->
    ``"Pocillopora grandis"``) buscándolo en ``codes`` (cargado desde
    ``cobertura_codes_v2.txt`` por :func:`load_coverage_codes`); si un
    código no tiene nombre registrado, se usa el código tal cual.

    Args:
        row: Fila (``pandas.Series`` o dict) con al menos las claves
            ``"Major Category"``, ``"Subcategory"``, ``"ID Code"``,
            ``"ID Name"`` y ``"Notes"``.
        codes: Diccionario de códigos devuelto por :func:`load_coverage_codes`.
        label_mode: ``"major"`` para usar solo la categoría mayor,
            ``"subcategory"`` para usar solo la subcategoría, o ``"full"``
            (por defecto) para combinar mayor + subcategoría + estado.
        sep: Separador usado entre las partes de la etiqueta compuesta.

    Returns:
        str: Etiqueta final lista para usarse como nombre de carpeta de
        clase (p.ej. ``"CORAL/POCILLOPORA_GRANDIS/CORAL_SANO"``), o
        ``"UNKNOWN"`` si no se pudo determinar ninguna parte.
    """
    major_raw = clean_cell(row.get("Major Category", ""))
    sub_raw = clean_cell(row.get("Subcategory", ""))
    id_code_raw = clean_cell(row.get("ID Code", ""))
    id_name_raw = clean_cell(row.get("ID Name", ""))
    notes_raw = clean_cell(row.get("Notes", ""))

    sub_code = slugify_label(sub_raw) if sub_raw else ""
    if not sub_code and id_code_raw:
        sub_code = slugify_label(id_code_raw)
    if not sub_code and id_name_raw:
        sub_code = slugify_label(id_name_raw)
    sub_code = SUBCATEGORY_MERGE.get(sub_code, sub_code)

    condition_code = slugify_label(notes_raw) if notes_raw else ""
    is_condition = condition_code in codes["state_codes"]

    parent_from_sub = codes["parent_by_subcategory"].get(sub_code, "")
    condition_points_to_coral = is_condition and parent_from_sub in ("", "CORAL")
    if condition_points_to_coral:
        major_code = "CORAL"
    else:
        major_code = parent_from_sub or code_from_major_value(major_raw, codes)
    if not major_code:
        major_code = "UNKNOWN"

    is_coral = major_code == "CORAL" or parent_from_sub == "CORAL"

    # Nombres legibles para mostrar en la etiqueta (los *_code siguen siendo
    # los códigos, usados arriba para la lógica de parentesco/condición).
    # slugify_label mantiene el nombre pero lo deja seguro para nombre de
    # carpeta (p.ej. "Pocillopora grandis" -> "POCILLOPORA_GRANDIS").
    major_name = slugify_label(codes["major_by_code"].get(major_code, major_code))
    sub_name = slugify_label(codes["subcategory_by_code"].get(sub_code, sub_code)) if sub_code else ""
    condition_name = slugify_label(codes["state_by_code"].get(condition_code, condition_code)) if condition_code else ""

    if label_mode == "major":
        parts = [major_name]
    elif label_mode == "subcategory":
        parts = [sub_name or major_name]
    else:
        parts = [major_name]
        if sub_name and sub_name != major_name:
            parts.append(sub_name)
        if is_coral and is_condition:
            parts.append(condition_name)

    return sep.join(p for p in parts if p) or "UNKNOWN"

def prepare_coords_labels(df, x_col, y_col, lbl_hint=None, label_mode="full", codes=None, label_sep="/", debug=False):
    """Extrae y normaliza las columnas x, y y etiqueta de un DataFrame de puntos.

    Si se indica ``lbl_hint`` (columna de etiqueta explícita), su valor se
    normaliza con :func:`slugify_label`. Si no, la etiqueta se construye
    fila a fila con :func:`build_composite_label`. Las filas cuyas
    coordenadas x/y no se pudieron convertir a número se descartan.

    Args:
        df (pandas.DataFrame): DataFrame con, al menos, las columnas de
            coordenadas indicadas en ``x_col``/``y_col``.
        x_col: Nombre de la columna con la coordenada X en píxeles.
        y_col: Nombre de la columna con la coordenada Y en píxeles.
        lbl_hint: Nombre exacto de la columna de etiqueta a usar
            directamente. Si es ``None``, se genera una etiqueta compuesta.
        label_mode: Modo de etiqueta compuesta cuando ``lbl_hint`` es
            ``None`` (ver :func:`build_composite_label`).
        codes: Diccionario de códigos de cobertura (ver
            :func:`load_coverage_codes`). Si es ``None``, se usa uno vacío.
        label_sep: Separador para las etiquetas compuestas.
        debug: Si es ``True``, imprime ejemplos de las etiquetas generadas.

    Returns:
        pandas.DataFrame: DataFrame con exactamente las columnas
        ``["x", "y", "label"]``, sin filas con coordenadas inválidas.

    Raises:
        ValueError: Si se indicó ``lbl_hint`` pero esa columna no existe
            en ``df``.
    """
    codes = codes or load_coverage_codes("")

    out = df.copy()
    out["x"] = normalize_numeric_series(out[x_col])
    out["y"] = normalize_numeric_series(out[y_col])

    if lbl_hint:
        if lbl_hint not in out.columns:
            raise ValueError(f"No encontré la columna de etiqueta indicada: '{lbl_hint}'. Cols: {list(out.columns)}")
        out["label"] = out[lbl_hint].map(slugify_label)
    else:
        out["label"] = out.apply(
            lambda r: build_composite_label(r, codes, label_mode=label_mode, sep=label_sep),
            axis=1,
        )

    out = out.dropna(subset=["x","y"]).reset_index(drop=True)
    if debug:
        examples = sorted(out["label"].dropna().unique())[:12]
        print(f"[DEBUG] etiquetas ejemplo ({label_mode}): {examples}")
    return out[["x","y","label"]]

def read_coords_labels_from_excel(path, sheet_hint="", x_col_hint=None, y_col_hint=None, lbl_hint=None, label_mode="full", codes=None, label_sep="/", debug=False):
    """Lee coordenadas y etiquetas desde un archivo Excel, eligiendo automáticamente la hoja correcta.

    Estrategia de selección de hoja (en orden):

    1. Si ``sheet_hint`` existe en el archivo y tiene columnas X/Y, se usa esa.
    2. Si no, se prioriza cualquier hoja cuyo nombre termine en ``"_archive"``
       y tenga columnas X/Y.
    3. Si no, se usa la primera hoja que tenga columnas X/Y.

    Una vez elegida la hoja, localiza las columnas de X, Y y (opcionalmente)
    etiqueta, y delega la limpieza final en :func:`prepare_coords_labels`.

    Args:
        path: Ruta al archivo ``.xlsx``/``.xls``.
        sheet_hint: Nombre de hoja preferido (opcional).
        x_col_hint: Nombre exacto de la columna X, si se conoce.
        y_col_hint: Nombre exacto de la columna Y, si se conoce.
        lbl_hint: Nombre exacto de la columna de etiqueta, si se conoce.
        label_mode: Modo de etiqueta compuesta (ver :func:`build_composite_label`).
        codes: Diccionario de códigos de cobertura (ver
            :func:`load_coverage_codes`).
        label_sep: Separador para las etiquetas compuestas.
        debug: Si es ``True``, imprime información de la hoja/columnas usadas.

    Returns:
        pandas.DataFrame: DataFrame con las columnas ``["x", "y", "label"]``.

    Raises:
        ValueError: Si ninguna hoja del Excel tiene columnas X/Y
            reconocibles, o si no se pueden identificar las columnas de
            coordenadas en la hoja seleccionada.
    """
    import pandas as pd
    from pathlib import Path

    xls = pd.ExcelFile(path)
    sheets = xls.sheet_names

    def sheet_has_xy(sh):
        tmp = pd.read_excel(path, sheet_name=sh)
        cols = [str(c).lower() for c in tmp.columns]
        has_x = any(c.strip().startswith("x") or "x (pixel" in c for c in cols)
        has_y = any(c.strip().startswith("y") or "y (pixel" in c for c in cols)
        return has_x and has_y, tmp

    # 1) Si pasas --sheet y existe, se usa (solo si tiene X/Y)
    df = None
    used_sheet = None
    if sheet_hint and sheet_hint in sheets:
        ok, tmp = sheet_has_xy(sheet_hint)
        if ok:
            df = tmp
            used_sheet = sheet_hint

    # 2) Intento auto: prioriza hojas que terminen en "_archive" con X/Y
    if df is None:
        for sh in sheets:
            if str(sh).lower().endswith("_archive"):
                ok, tmp = sheet_has_xy(sh)
                if ok:
                    df = tmp
                    used_sheet = sh
                    break

    # 3) Fallback auto: cualquier hoja con X/Y
    if df is None:
        for sh in sheets:
            ok, tmp = sheet_has_xy(sh)
            if ok:
                df = tmp
                used_sheet = sh
                break

    if df is None:
        raise ValueError(f"Excel '{Path(path).name}': no encontré ninguna hoja con columnas X/Y. Hojas: {sheets}")

    if debug:
        print(f"[DEBUG] Excel {Path(path).name}: hoja usada='{used_sheet}' columnas: {list(df.columns)}")

    x_col = x_col_hint or find_col(df.columns, ["x (pixel","x_pixel","xpixel","x [pix","x pix"," x","x)","x"])
    y_col = y_col_hint or find_col(df.columns, ["y (pixel","y_pixel","ypixel","y [pix","y pix"," y","y)","y"])
    if x_col is None and "X" in df.columns: x_col = "X"
    if y_col is None and "Y" in df.columns: y_col = "Y"
    if x_col is None or y_col is None:
        raise ValueError(f"Excel '{Path(path).name}': no encontré columnas X/Y en hoja '{used_sheet}'. Cols: {list(df.columns)}")

    out = prepare_coords_labels(
        df,
        x_col,
        y_col,
        lbl_hint=lbl_hint,
        label_mode=label_mode,
        codes=codes,
        label_sep=label_sep,
        debug=debug,
    )

    if debug:
        label_desc = lbl_hint if lbl_hint else f"compuesta:{label_mode}"
        print(f"[DEBUG] filas válidas={len(out)} (sheet='{used_sheet}', x='{x_col}', y='{y_col}', label='{label_desc}')")
    return out


def clamp(v, lo, hi):
    """Restringe un valor al rango cerrado [lo, hi].

    Args:
        v: Valor a restringir.
        lo: Límite inferior.
        hi: Límite superior.

    Returns:
        ``v`` si ya está dentro de ``[lo, hi]``; en caso contrario, el
        límite más cercano.
    """
    return max(lo, min(hi, v))

def crop_centered_patch(im, cx, cy, size=40, pad_edge=True):
    """Recorta un parche cuadrado de la imagen centrado en (cx, cy).

    Si el recorte se sale de los bordes de la imagen, se recorta primero
    dentro de los límites válidos y, si ``pad_edge`` es ``True``, el parche
    resultante se pega sobre un lienzo negro del tamaño solicitado para
    mantener siempre la misma resolución de salida.

    Args:
        im (PIL.Image.Image): Imagen fuente (se recomienda en modo "RGB").
        cx: Coordenada X (en píxeles) del centro del parche.
        cy: Coordenada Y (en píxeles) del centro del parche.
        size: Lado del parche cuadrado de salida, en píxeles.
        pad_edge: Si es ``True``, rellena con negro los parches que caen
            parcialmente fuera de la imagen para que siempre midan
            ``size x size``. Si es ``False``, se devuelve el recorte tal
            cual (puede ser más pequeño en los bordes).

    Returns:
        PIL.Image.Image: El parche recortado (y opcionalmente rellenado).
    """
    W,H = im.size; half = size//2
    x0,y0 = int(round(cx-half)), int(round(cy-half))
    x1,y1 = int(round(cx+half)), int(round(cy+half))
    ix0,iy0 = clamp(x0,0,W), clamp(y0,0,H)
    ix1,iy1 = clamp(x1,0,W), clamp(y1,0,H)
    patch = im.crop((ix0,iy0,ix1,iy1))
    if not pad_edge or patch.size==(size,size): return patch
    canvas = Image.new("RGB",(size,size),(0,0,0))
    canvas.paste(patch,(ix0-x0,iy0-y0))
    return canvas

def ensure_dir(p:Path):
    """Crea un directorio (y sus padres) si no existe todavía.

    Args:
        p (Path): Ruta del directorio a crear.
    """
    p.mkdir(parents=True, exist_ok=True)

def main():
    """Punto de entrada del script.

    Procesa los argumentos de línea de comandos, empareja cada imagen de
    ``--images_dir`` con su Excel/CSV homónimo en ``--excels_dir``, calcula
    un split train/val/test por imagen (para evitar fuga de datos entre
    parches de la misma foto), recorta un parche por cada punto etiquetado
    y guarda todo en ``--out_dir/<patch_size>x<patch_size>/<split>/<clase>/``.

    No recibe argumentos ni devuelve nada directamente: toda la
    configuración se lee de ``sys.argv`` mediante ``argparse`` (ver
    ``python scripts/batch_crop_and_label_from_excel.py --help`` para la
    lista completa de opciones).
    """
    ap = argparse.ArgumentParser()
    ap.add_argument("--images_dir", required=True)
    ap.add_argument("--excels_dir", required=True)
    ap.add_argument("--out_dir", default="./dataset")
    ap.add_argument("--sheet", default="", help="Hoja del Excel (p.ej. DSCN9411_archive)")
    ap.add_argument("--x_col", default="", help="Nombre exacto columna X (opcional)")
    ap.add_argument("--y_col", default="", help="Nombre exacto columna Y (opcional)")
    ap.add_argument("--label_col", default="", help="Columna de etiqueta exacta. Si se deja vacío, crea etiqueta compuesta")
    ap.add_argument("--label_mode", choices=["major", "subcategory", "full"], default="full",
                    help="major=solo categoría mayor; subcategory=solo subcategoría; full=mayor+subcategoría+estado coral")
    ap.add_argument("--label_sep", default="/", help="Separador para etiquetas compuestas (usa '/' para crear subcarpetas categoria/subcategoria)")
    ap.add_argument("--codes_file", default="cobertura_codes_v3.json", help="Archivo con códigos de cobertura (.json v3 o .txt v2)")
    ap.add_argument("--patch_size", type=int, default=40)
    ap.add_argument("--pad_edge", type=str, default="true")
    ap.add_argument("--scale_x", type=float, default=1.0)
    ap.add_argument("--scale_y", type=float, default=1.0)
    ap.add_argument("--val_ratio", type=float, default=0.15)
    ap.add_argument("--test_ratio", type=float, default=0.15)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--debug", type=str, default="false")
    args = ap.parse_args()

    pad_edge = args.pad_edge.lower() in ("true","1","yes","y","t")
    debug = args.debug.lower() in ("true","1","yes","y","t")
    random.seed(args.seed)

    images_dir = Path(args.images_dir)
    excels_dir = Path(args.excels_dir)
    codes_file = Path(args.codes_file)
    if not codes_file.is_absolute():
        codes_file = Path.cwd() / codes_file
    coverage_codes = load_coverage_codes(codes_file, debug=debug)

    # Carpeta raíz según tamaño de parche, p.ej. ./datasets/32x32
    root_out_dir = Path(args.out_dir) / f"{args.patch_size}x{args.patch_size}"
    out_dir = root_out_dir

    ensure_dir(root_out_dir)

    # recolecta pares imagen-excel homónimos
    images = sorted(
    [p for p in images_dir.iterdir() if p.suffix.lower() in IMG_EXTS],
    key=lambda p: p.name.lower()
    )
    pairs = []
    for img in images:
        stem = img.stem
        excel = None
        for ext in XLS_EXTS + CSV_EXTS:
            cand = excels_dir / f"{stem}{ext}" 
            if cand.exists():
                excel = cand; break
        if excel: pairs.append((img, excel))
        elif debug:
            print(f"[AVISO] No encontré Excel/CSV para {img.name}")

    # split por imagen (evita fuga)
    random.shuffle(pairs)
    n = len(pairs)
    n_test = int(round(n*args.test_ratio))
    n_val  = int(round(n*args.val_ratio))
    test_pairs  = pairs[:n_test]
    val_pairs   = pairs[n_test:n_test+n_val]
    train_pairs = pairs[n_test+n_val:]

    def split_of(img_path):
        if img_path in set(p[0] for p in test_pairs): return "test"
        if img_path in set(p[0] for p in val_pairs):  return "val"
        return "train"

    total = 0
    for img_path, excel_path in pairs:
        # --- CSV: X/Y + etiqueta compuesta desde Major/Subcategory/Notes ---
        if excel_path.suffix.lower() in CSV_EXTS:
            df = pd.read_csv(excel_path)

            # nombres esperados
            xcol = args.x_col or "X"
            ycol = args.y_col or "Y"

            # por si acaso, si no están exactos, intenta encontrarlos por nombre (case-insensitive)
            if xcol not in df.columns:
                xcol = find_col(df.columns, ["x"])
            if ycol not in df.columns:
                ycol = find_col(df.columns, ["y"])
            lcol = args.label_col or ""
            if lcol and lcol not in df.columns:
                lcol = find_col(df.columns, [lcol])

            if xcol is None or ycol is None or xcol not in df.columns or ycol not in df.columns:
                print(f"[ERROR] {excel_path.name}: necesito columnas 'X' y 'Y'")
                continue
            if args.label_col and (lcol is None or lcol not in df.columns):
                print(f"[ERROR] {excel_path.name}: no encontré la columna de etiqueta '{args.label_col}'")
                continue

            df = prepare_coords_labels(
                df,
                xcol,
                ycol,
                lbl_hint=(lcol or None),
                label_mode=args.label_mode,
                codes=coverage_codes,
                label_sep=args.label_sep,
                debug=debug,
            )

            if debug:
                label_desc = lcol if lcol else f"compuesta:{args.label_mode}"
                print(f"[DEBUG] CSV {excel_path.name}: filas válidas={len(df)} (x='{xcol}', y='{ycol}', label='{label_desc}')")

        # --- Excels normales ---
        else:
            df = read_coords_labels_from_excel(
                excel_path,
                sheet_hint=(args.sheet or ""),
                x_col_hint=(args.x_col or None),
                y_col_hint=(args.y_col or None),
                lbl_hint=(args.label_col or None),
                label_mode=args.label_mode,
                codes=coverage_codes,
                label_sep=args.label_sep,
                debug=debug,
            )

        if df.empty:
            if debug: print(f"[AVISO] {excel_path.name}: sin filas válidas")
            continue

        im = Image.open(img_path).convert("RGB")
        subset = split_of(img_path)

        for idx, r in df.iterrows():
            cx = float(r["x"])*args.scale_x
            cy = float(r["y"])*args.scale_y
            label = str(r.get("label","unknown")).strip() or "unknown"

            # carpeta destino
            dst = out_dir / subset / label
            ensure_dir(dst)

            patch = crop_centered_patch(im, cx, cy, size=args.patch_size, pad_edge=pad_edge)
            name = f"{img_path.stem}_p{idx+1:03d}_x{int(round(cx))}_y{int(round(cy))}.jpg"
            patch.save(dst / name, quality=95)
            total += 1

    print(f"Listo. Parches totales: {total}. Estructura en: {out_dir}")
    print("Ej.: dataset/train/<clase>/*.jpg, dataset/val/<clase>/*.jpg, dataset/test/<clase>/*.jpg")

if __name__ == "__main__":
    main()
