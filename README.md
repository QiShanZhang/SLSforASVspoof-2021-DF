Audio Deepfake Detection with XLS-R and SLS classfier
===============
This repository contains our implementation of the paper "Audio Deepfake Detection with XLS-R and SLS classfier  Qishan Zhang, Shuangbing Wen, Tao Hu ACM MM 2024"（https://openreview.net/pdf?id=acJMIXJg2u){https://openreview.net/pdf}



## Installation
First, clone the repository locally, create and activate a conda environment, and install the requirements :
```
$ git clone 
$ conda create -n SLS python=3.7
$ conda activate SLS
$ pip install torch==1.12.1 torchvision==0.13.1 torchaudio==0.12.1
$ cd fairseq-a54021305d6b3c4c5959ac9395135f63202db8f1
$ pip install --editable ./
$ cd ..
$ pip install -r requirements.txt
```


## Experiments

### Dataset
Our experiments are performed on the public dataset logical access (LA) and deepfake (DF) partition of the ASVspoof 2021 dataset and In-the-Wild dataset(train on 2019 LA training and evaluate on 2021 LA and DF, In-the-Wild evaluation database).

The ASVspoof 2019 dataset, which can can be downloaded from [here](https://datashare.is.ed.ac.uk/handle/10283/3336).

The ASVspoof 2021 database is released on the zenodo site.

LA [here](https://zenodo.org/record/4837263#.YnDIinYzZhE)

DF [here](https://zenodo.org/record/4835108#.YnDIb3YzZhE)

The In-the-Wild dataset can be downloaded from [here](https://deepfake-total.com/in_the_wild)

For ASVspoof 2021 dataset keys (labels) and metadata are available [here](https://www.asvspoof.org/index2021.html)

## Pre-trained wav2vec 2.0 XLS-R (300M)
Download the XLS-R models from [here](https://github.com/pytorch/fairseq/tree/main/examples/wav2vec/xlsr)

## Training model
To train the model run:
```
CUDA_VISIBLE_DEVICES=0 python main.py --track=DF --lr=0.000001 --batch_size=5 --loss=WCE  --num_epochs=50
```
## Testing

To evaluate your own model on the DF, LA, and In-the-Wild evaluation datasets: The code below will generate three 'score.txt' files, one for each evaluation dataset, and these files will be used to compute the EER(%).
```
CUDA_VISIBLE_DEVICES=0 python main.py   --track=DF --is_eval --eval 
                                        --model_path=/path/to/your/best_model.pth
                                        --protocols_path=database/ASVspoof_DF_cm_protocols/ASVspoof2021.DF.cm.eval.trl.txt 
                                        --database_path=/path/to/your/ASVspoof2021_DF_eval/ 
                                        --eval_output=/path/to/your/scores_DF.txt

CUDA_VISIBLE_DEVICES=0 python main.py   --track=LA --is_eval --eval 
                                        --model_path=/path/to/your/best_model.pth
                                        --protocols_path=database/ASVspoof_DF_cm_protocols/ASVspoof2021.LA.cm.eval.trl.txt 
                                        --database_path=/path/to/your/ASVspoof2021_LA_eval/ 
                                        --eval_output=/path/to/your/scores_LA.txt

CUDA_VISIBLE_DEVICES=0 python main.py   --track=In-the-Wild --is_eval --eval 
                                        --model_path=/path/to/your/best_model.pth
                                        --protocols_path=database/ASVspoof_DF_cm_protocols/in_the_wild.eval.txt 
                                        --database_path=/path/to/your/release_in_the_wild/ 
                                        --eval_output=/path/to/your/scores_In-the-Wild.txt
```
We also provide a pre-trained model. To use it, you can download from [here](https://drive.google.com/drive/folders/13vw_AX1jHdYndRu1edlgpdNJpCX8OnrH?usp=sharing) and change the --model_path to our pre-trained model.

Compute the EER(%) use three 'scores.txt' file
```
python evaluate_2021_DF.py scores/scores_DF.txt ./keys eval

python evaluate_2021_LA.py scores/scores_LA.txt ./keys eval

python evaluate_in_the_wild.py scores/scores_Wild.txt ./keys eval
``` 

## Results using pre-trained model:
EER: 1.92 % on ASVspoof 2021 DF dataset.
EER: 2.87 % on ASVspoof 2021 LA dataset.
EER: 7.46 % on In-the-Wild dataset.


