<div align="center">
<h2>MECAM-OOD</h2>

[Zixuan Pan](https://scholar.google.com/citations?user=3VuW2gcAAAAJ&hl=en), [Yu-Jen Chen](https://scholar.google.com/citations?hl=en&user=88f7rGMAAAAJ), [Jun Xia](https://scholar.google.com/citations?hl=en&user=K4JXEHUAAAAJ), [Max Ficco](https://maxficco.com), [Jianxu Chen](https://scholar.google.com/citations?user=HdolpOgAAAAJ&hl=en), [Tsung-Yi Ho](https://scholar.google.com/citations?hl=en&user=TRDUYkAAAAAJ), [Yiyu Shi](https://scholar.google.com/citations?hl=en&user=LrjbEkIAAAAJ&view_op=list_works)

</div>

## Dataset

### ID Dataset - ISIC19, PathMNIST
- [ISIC 2019](https://challenge.isic-archive.com/data/#2019)
- PathMNIST (from medmnist import PathMNIST)

```
DATASET_NAME
|-- train
|   |-- class_1
|   |   |-- xxx.jpg
|   |   |-- ...
|   |-- class_2
|   |-- class_3
|   |-- ...
|-- test
|   |-- class_1
|   |   |-- xxx.jpg
|   |   |-- ...
|   |-- class_2
|   |-- class_3
|   |-- ...
```
### OOD Dataset - RSNA, COVID-19, HeadCT
- [RSNA](https://www.kaggle.com/competitions/rsna-pneumonia-detection-challenge/data)
- [COVID-19](https://www.kaggle.com/datasets/tawsifurrahman/covid19-radiography-database)
- [HeadCT](https://www.kaggle.com/datasets/felipekitamura/head-ct-hemorrhage)
```
DATASET_NAME
|-- class_1
|   |-- xxx.jpg
|   |-- ...
|-- class_2
|-- class_3
|-- ...
```

## Train Classification Model with ID Dataset
```
cd CNet_ME
python3 train_cnet.py -b 128 -d ISIC --model_type Res18 -e 200
```

## Run MECAM-OOD
```
cd MECAM-OODD
python3 main.py -id ISIC -ood RSNA --pretrained_path MERes18_ISIC_1231_003419 -gid 0
```


