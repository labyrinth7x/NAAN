# Semi-supervised-face-recognition
![fw](https://github.com/labyrinth7x/Semi-supervised-face-recognition/blob/master/resources/fw.png)


## Introduction
This repository is for our IJCNN'20 paper [Neighborhood-Aware Attention Network for
Semi-supervised Face Recognition](). (will be released **soon**)

## Requirements
- Python 3.5
- Pytorch 1.0.0
- numpy
- faiss

## Data Preparation
- Download the full MS-Celeb-1M realeased by [ArcFace](https://github.com/deepinsight/insightface) from [baidu](https://pan.baidu.com/s/1S6LJZGdqcZRle1vlcMzHOQ) or [dropbox](https://www.dropbox.com/s/wpx6tqjf0y5mf6r/faces_ms1m-refine-v2_112x112.zip?dl=0).
- Download the splitted image list produced by [learn-to-cluster](https://github.com/yl-1993/learn-to-cluster) from [GoogleDrive](https://drive.google.com/file/d/1kurPWh6dm3dWQOLqUAeE-fxHrdnjaULB/view?usp=sharing) or [OneDrive](https://mycuhk-my.sharepoint.com/:u:/g/personal/1155095455_link_cuhk_edu_hk/ET7lHxOXSjtDiMsgqzLK9LgBi_QW0WVzgZdv2UBzE1Bgzg?e=jZ7kCS).
- Download the extracted features and precomputed knn for split0 (labeled) from [GoogleDrive](https://drive.google.com/open?id=1BIij-1kQ2OybcmUqRhLyaAfRb1MsE9He) and move them to ```data/labeled```.
- Download the extracted features and precomputed knn for split1 (unlabeled) from [GoogleDrive](https://drive.google.com/drive/folders/1zBmoowfo-eMloo9iXY3TMyIRecB0Uhvt?usp=sharing) and move them to ```data/unlabeled```. Or you can download the pretrained model face recognition model from [GoogleDrive](), and extract/compute the features and knn files and move them to the same directory.


## Training
```
sh train_gat.sh
```
We only use **part0_train.list** to train the GAT and classifier.  
You can also download the pre-trained model weights from [GoogleDrive](https://drive.google.com/file/d/1xKbYTF_Q3IC8mOkmMpvzR4VC9vzKb9En/view?usp=sharing)．

## Evaluation
- Generate pseudo labels for unlabeled data
  ```
  sh eval.sh
  ```
  You should specify the params ```model_path``` or use the default setting.   
  Or you can use your own data by preparing ```knn,features,labels``` files like in ```data/unlabeled``` and specify the params ```knn_path``` and ```feat_path``` in eval_gat.py.

- Evaluate with the true labels
  ```
  python statistics.py {pseudo_label_dir}
  ```
  You should specify the pseudo_label_dir as command-line arguments.  
  It will output the BCubed F-measure, and restore the new pseudo_label_clean file by removing the singleton clusters in the same pseudo_label_dir.
 
 ## Joint Training Framework
 It will be released soon.
