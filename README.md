# XLTime
This repository contains the source code, preprocessed dataset, and reproduction scripts for the NAACL 2022 paper [XLTime: A Cross-Lingual Knowledge Transfer Framework for Temporal Expression Extraction](https://openreview.net/pdf?id=6dXfj57KVdp).

./XLTime contains the source code of the proposed model.

./data contains the preprocessed data needed to reproduce the XLTime experiments in the paper.

./reproduce_results.sh contains the scripts needed to reproduce the XLTime experiments in the paper.

# To run XLTime
```
# This example shows how to apply XLTime on the mBERT backbone and transfer from EN to FR.
# Running this example would present XLTime-mBERT (transfer from EN) result for FR TEE, as
# shown in row 9, column 1 of Table 3 and row 9, column 1 of the upper as well as the lower
# part of Table 8 of the paper.
# To reproduce the other XLTime experiments in the paper, please see reproduce_results.sh.

# get 'w/ type' result (row 9, column 1 of the upper part of Table 8 of the paper)
python ./XLTime/main.py --data_dir_sl=./data/sl_EN_2_FR/ --data_dir_bc=./data/bc_EN_2_FR/ --output_dir=./XLTime-mBERT_EN_2_FR_results/ --num_train_epochs=50 --do_train --do_eval --warmup_proportion=0.5 --learning_rate=0.000007 --train_batch_size=4 --dropout=0.2 --backbone=mBERT --model_size=base

# map the result to 'w/o type' result (shown in row 9, column 1 of Table 3 and row 9, 
# column 1 of the lower part of Table 8 of the paper)
python ./XLTime/map_results.py --data_path ./XLTime-mBERT_EN_2_FR_results/
```
# Citation
If you find this repository helpful, please consider citing out paper:
```bibtex
@inproceedings{cao2022xltime,
  title={XLTime: A Cross-Lingual Knowledge Transfer Framework for Temporal Expression Extraction},
  author={Yuwei Cao and William Groves and Tanay Kumar Saha and Joel R. Tetreault and Alex Jaimes and Hao Peng and Philip S. Yu},
  booktitle={2022 Annual Conference of the North American Chapter of the Association for Computational Linguistics},
  url={https://openreview.net/forum?id=6dXfj57KVdp},
  year={2022}
}
```
