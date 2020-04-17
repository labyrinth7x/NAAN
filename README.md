# Neighborhood-Aware Attention Network for Semi-supervised Face Recognition
This repository is the official implementation of our IJCNN'20 paper [Neighborhood-Aware Attention Network for
Semi-supervised Face Recognition](https://drive.google.com/file/d/1fNarQTLGRcmf06C1Uhytjcbn3U9hknf0/view?usp=sharing).  
<img src="https://github.com/labyrinth7x/Semi-supervised-face-recognition/blob/master/resources/fw.png" width = 80% height = 80% div align=center />

## Requirements
* Python >= 3.5
* PyTorch >= 1.0.0
* torchvision >= 0.2.1
* numpy
* tqdm

To install requirements:
```
python install -r requirements.txt
```


## Data Preparation
- Download the full MS-Celeb-1M realeased by [ArcFace](https://github.com/deepinsight/insightface) from [baidu](https://pan.baidu.com/s/1S6LJZGdqcZRle1vlcMzHOQ) or [dropbox](https://www.dropbox.com/s/wpx6tqjf0y5mf6r/faces_ms1m-refine-v2_112x112.zip?dl=0).
- Download the splitted image list produced by [learn-to-cluster](https://github.com/yl-1993/learn-to-cluster) from [GoogleDrive](https://drive.google.com/file/d/1kurPWh6dm3dWQOLqUAeE-fxHrdnjaULB/view?usp=sharing) or [OneDrive](https://mycuhk-my.sharepoint.com/:u:/g/personal/1155095455_link_cuhk_edu_hk/ET7lHxOXSjtDiMsgqzLK9LgBi_QW0WVzgZdv2UBzE1Bgzg?e=jZ7kCS).
- Download the extracted features and precomputed knn for split0 (labeled) from [GoogleDrive](https://drive.google.com/open?id=1BIij-1kQ2OybcmUqRhLyaAfRb1MsE9He) and move them to ```data/labeled```.
- Download the extracted features and precomputed knn for split1 (unlabeled) from [GoogleDrive](https://drive.google.com/drive/folders/1zBmoowfo-eMloo9iXY3TMyIRecB0Uhvt?usp=sharing) and move them to ```data/unlabeled```. We also provide the pretrained face recognition model (ArcFace trained using only split0) from [GoogleDrive](). You can use it to extract/compute features and knn files for your own data and move them to ```data/unlabeled```. The structure of ```data``` is the same as:
  ```
  data
   ├── labeled
      ├── split0_feats.npz
      ├── split0_knn.npy
   ├── unlabeled
      ├── split1_feats.npz
      ├── split1_knn.npy
      ├── split1_labels.txt
  ```

## Training
```
sh train_gat.sh
```
Please note that:  
* We only use **part0_train.list** to train the GAT and classifier.  
* You can download the pre-trained model weights from [GoogleDrive](https://drive.google.com/file/d/1xKbYTF_Q3IC8mOkmMpvzR4VC9vzKb9En/view?usp=sharing)．

## Evaluation

1. Generate pseudo labels for unlabeled data
    ```
    sh eval.sh
    ```
    Please note that:  
    * You should change the param ```model_path``` with your own or use the default setting.   
    * After preparing your own data, you can change the params ```knn_path``` and ```feat_path``` in ```eval_gat.py``` to generate pseudo labels.

2. Evaluate with the true labels
    ```
    python statistics.py \
    pseudo_dir pseudl_data \
    split split1
    ```
    Please note that:  
    * If your dirname of the generated pseudo label file is different, please replace ```pseudl_data``` with your own.
    * If your test set is not ```split1```, please change ```split``` to your own data split.
    * It will output results, including BCubed Precision, BCubed Recall and BCubed F-measure.
    * The new pseudo_label_clean file will be saved into ```pseudo_dir``` by removing singleton clusters.
 
 ## Joint Training Framework
 You may refer to the following repository:  
 https://github.com/labyrinth7x/multi-task-face-recognition-framework
 
 ## Results on split1_test
 Method | Precision | Recall | F-score  
-|-|-|-
NAAN | 97.0 | 86.8 | 91.6

After removing singleton clusters
Method | Precision | Recall | F-score | Discard ratio (%)
-|-|-|-|-
NAAN | 96.9 | 93.6 | 95.2 | 4.2 |
 
 
 
 ## Citation
 Please cite our paper if it helps your research:
 ```
 @inproceedings{DBLP:conf/ijcnn/ZhangLL19,
  author    = {Qi Zhang, Zhen Lei, Stan Z.Li},
  title     = {Neighborhood-Aware Attention Network for Semi-supervised Face Recognition},
  booktitle = {IJCNN},
  year      = {2020}
  ```
   
  ## Ackownledgement
  The codes for pseudo label propagation are from [CDP](https://github.com/XiaohangZhan/cdp). Thanks for their work.
