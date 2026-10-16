"""Cuenta parches por clase (mayor / especie / condición) y split en un dataset recortado."""
import os, sys, collections
from pathlib import Path
sys.stdout.reconfigure(encoding="utf-8")
root = Path(sys.argv[1])
rows = []
for split in ("train", "val", "test"):
    for dp, _, fns in os.walk(root / split):
        n = sum(f.lower().endswith((".jpg", ".jpeg", ".png")) for f in fns)
        if n:
            rows.append((split, Path(dp).relative_to(root / split).parts, n))
def table(keyf, title):
    c = collections.defaultdict(collections.Counter)
    for s, parts, n in rows:
        k = keyf(parts)
        if k: c[k][s] += n
    tot = sum(sum(v.values()) for v in c.values())
    print(f"\n### {title}\n| Clase | train | val | test | total | % |\n|---|---|---|---|---|---|")
    for k, v in sorted(c.items(), key=lambda kv: -sum(kv[1].values())):
        t = sum(v.values()); print(f"| {k} | {v['train']} | {v['val']} | {v['test']} | {t} | {100*t/tot:.2f} |")
    print(f"| **Total** | {sum(v['train'] for v in c.values())} | {sum(v['val'] for v in c.values())} | {sum(v['test'] for v in c.values())} | {tot} | 100 |")
table(lambda p: p[0], "Categoría mayor")
table(lambda p: "/".join(p[:2]), "Categoría/subcategoría (especie)")
table(lambda p: p[2] if p[0] == "CORAL" and len(p) >= 3 else None, "Condición (solo CORAL)")
table(lambda p: "/".join(p), "Hoja completa (categoría/subcategoría/condición)")
