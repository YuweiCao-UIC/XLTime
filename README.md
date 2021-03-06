# XLTime
This repository contains the source code, preprocessed dataset, and reproduction scripts for the NAACL 2022 paper [XLTime: A Cross-Lingual Knowledge Transfer Framework for Temporal Expression Extraction](https://arxiv.org/abs/2205.01757).

./XLTime contains the source code of the proposed model.

./data contains the preprocessed data needed to reproduce the XLTime experiments in the paper.

./reproduce_results.sh contains the scripts needed to reproduce the XLTime experiments in the paper.

We also provide a sample trained model in [this Google drive folder](https://drive.google.com/drive/folders/1tRBwA4ABhJvsoF2cIw9QJU6NMwshNESl?usp=sharing), see the 'To load a trained XLTime model' section below for more details. 


# To run XLTime
```
# This example shows how to apply XLTime on the mBERT backbone and transfer from EN to FR.
# Running this example would present XLTime-mBERT (transfer from EN) result for FR TEE, as
# shown in row 9, column 1 of Table 3 and row 9, column 1 of the upper as well as the lower
# part of Table 8 of the paper.
# To reproduce the other XLTime experiments in the paper, please see reproduce_results.sh.

pip install -r XLTime/requirements.txt

# get 'w/ type' result (row 9, column 1 of the upper part of Table 8 of the paper)
python ./XLTime/main.py --data_dir_sl=./data/sl_EN_2_FR/ --data_dir_bc=./data/bc_EN_2_FR/ --output_dir=./XLTime-mBERT_EN_2_FR_results/ --num_train_epochs=50 --do_train --do_eval --warmup_proportion=0.5 --learning_rate=0.000007 --train_batch_size=4 --dropout=0.2 --backbone=mBERT --model_size=base

# map the result to 'w/o type' result (shown in row 9, column 1 of Table 3 and row 9, 
# column 1 of the lower part of Table 8 of the paper)
python ./XLTime/map_results.py --data_path ./XLTime-mBERT_EN_2_FR_results/
```
# To load a trained XLTime model
```
# We provide a sample trained model at https://drive.google.com/drive/folders/1tRBwA4ABhJvsoF2cIw9QJU6NMwshNESl?usp=sharing
# This sample model is trained by applying XLTime on mBERT and transferring knowledge 
# from English to French for French temporal expression extraction.
# To load and evaluate it, download ./XLTime-mBERT_EN_2_FR_results/ to the root folder and run:

# get 'w/ type' result (row 9, column 1 of the upper part of Table 8 of the paper)
python ./XLTime/main.py --data_dir_sl=./data/sl_EN_2_FR/ --data_dir_bc=./data/bc_EN_2_FR/ --output_dir=./XLTime-mBERT_EN_2_FR_results/ --do_eval --backbone=mBERT --model_size=base

# map the result to 'w/o type' result (shown in row 9, column 1 of Table 3 and row 9, 
# column 1 of the lower part of Table 8 of the paper)
python ./XLTime/map_results.py --data_path ./XLTime-mBERT_EN_2_FR_results/
```
# Citation
If you find this repository helpful, please consider citing our paper:
```bibtex
@inproceedings{cao2022xltime,
  title={XLTime: A Cross-Lingual Knowledge Transfer Framework for Temporal Expression Extraction},
  author={Yuwei Cao and William Groves and Tanay Kumar Saha and Joel R. Tetreault and Alex Jaimes and Hao Peng and Philip S. Yu},
  booktitle={the Findings of NAACL 2022},
  url={https://openreview.net/forum?id=6dXfj57KVdp},
  year={2022}
}
```
