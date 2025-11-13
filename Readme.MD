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
python scripts/batch_crop_and_label_from_excel.py \
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

python scripts/train_mlp_patches.py \
  --data_dir ./datasets/32x32 \
  --patch_size 32 \
  --epochs 25 \
  --batch_size 64 \
  --lr 1e-3 \
  --hidden "1024,512" \
  --dropout 0.3
```

## Correr CNN simple
```bash
python scripts/train_cnn_patches.py \
  --data_dir ./datasets/32x32 \
  --patch_size 32 \
  --epochs 40 \
  --batch_size 32 \
  --lr 3e-4 \
  --dropout 0.3
```

## Fine-tuning de modelo resnet18
```bash
python train_finetune.py \
  --data_dir ./datasets/32x32 \
  --model resnet18 \
  --epochs 20 \
  --batch_size 32 \
  --lr 1e-4
```

## Fine-tuning de modelo vgg16 
```bash
python train_finetune.py \
  --data_dir ./datasets/32x32 \
  --model vgg16  \
  --epochs 20 \
  --batch_size 32 \
  --lr 1e-4
```

## Fine-tuning de modelo AlexNet
```bash
python train_finetune.py \
  --data_dir ./datasets/32x32 \
  --model AlexNet \
  --epochs 20 \
  --batch_size 32 \
  --lr 1e-4
```

## Fine-tuning de modelo MobileNetV2
```bash
python train_finetune.py \
  --data_dir ./datasets/32x32 \
  --model MobileNetV2 \
  --epochs 20 \
  --batch_size 32 \
  --lr 1e-4
```
