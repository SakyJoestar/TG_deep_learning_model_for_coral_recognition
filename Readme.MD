## Librerias

```bash
pip install requirements.txt
```

## Comando para parches
```bash
python batch_crop_from_excel_per_image.py \
  --images_dir ./images \
  --excels_dir ./excels \
  --out_dir ./patches \
  --patch_size 40 \
  --pad_edge true
```

## Parches y etiquetas desde excel
```bash 
python batch_crop_and_label_from_excel.py \
  --images_dir "./data/imagenes" \
  --excels_dir "./data/csvs" \
  --out_dir "./datasets" \
  --patch_size 32 \
  --val_ratio 0.15 \
  --test_ratio 0.15 \
  --seed 42 \
  --debug true
```

## Correr percepton multicapa
```bash

python train_mlp_patches.py \
  --data_dir ./dataset \
  --epochs 25 \
  --batch_size 64 \
  --lr 1e-3 \
  --hidden "1024,512" \
  --dropout 0.3
```

