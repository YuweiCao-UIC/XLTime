# This file contains the scripts needed to reproduce the XLTime experiments in the paper.

# XLTime-mBERT, transfer from EN to FR 
# get 'w/ type' result (row 9, column 1 of the upper part of Table 8 of the paper)
python ./XLTime/main.py --data_dir_sl=./data/sl_EN_2_FR/ --data_dir_bc=./data/bc_EN_2_FR/ --output_dir=./XLTime-mBERT_EN_2_FR_results/ --num_train_epochs=50 --do_train --do_eval --warmup_proportion=0.5 --learning_rate=0.000007 --train_batch_size=4 --dropout=0.2 --backbone=mBERT --model_size=base
# map the result to 'w/o type' result (shown in row 9, column 1 of Table 3 and row 9, 
# column 1 of the lower part of Table 8 of the paper)
python ./XLTime/map_results.py --data_path ./XLTime-mBERT_EN_2_FR_results/


# XLTime-mBERT, transfer from EN to ES
# get 'w/ type' result (row 9, column 2 of the upper part of Table 8 of the paper)
python ./XLTime/main.py --data_dir_sl=./data/sl_EN_2_ES/ --data_dir_bc=./data/bc_EN_2_ES/ --output_dir=./XLTime-mBERT_EN_2_ES_results/ --num_train_epochs=50 --do_train --do_eval --warmup_proportion=0.5 --learning_rate=0.000007 --train_batch_size=4 --dropout=0.2 --backbone=mBERT --model_size=base
# map the result to 'w/o type' result (shown in row 9, column 2 of Table 3 and row 9, 
# column 2 of the lower part of Table 8 of the paper)
python ./XLTime/map_results.py --data_path ./XLTime-mBERT_EN_2_ES_results/


# XLTime-mBERT, transfer from EN to PT
# get 'w/ type' result (row 9, column 3 of the upper part of Table 8 of the paper)
python ./XLTime/main.py --data_dir_sl=./data/sl_EN_2_PT/ --data_dir_bc=./data/bc_EN_2_PT/ --output_dir=./XLTime-mBERT_EN_2_PT_results/ --num_train_epochs=50 --do_train --do_eval --warmup_proportion=0.5 --learning_rate=0.000007 --train_batch_size=4 --dropout=0.2 --backbone=mBERT --model_size=base
# map the result to 'w/o type' result (shown in row 9, column 3 of Table 3 and row 9, 
# column 3 of the lower part of Table 8 of the paper)
python ./XLTime/map_results.py --data_path ./XLTime-mBERT_EN_2_PT_results/


# XLTime-mBERT, transfer from EN to EU
# get 'w/ type' result (row 9, column 4 of the upper part of Table 8 of the paper)
python ./XLTime/main.py --data_dir_sl=./data/sl_EN_2_EU/ --data_dir_bc=./data/bc_EN_2_EU/ --output_dir=./XLTime-mBERT_EN_2_EU_results/ --num_train_epochs=50 --do_train --do_eval --warmup_proportion=0.5 --learning_rate=0.000007 --train_batch_size=4 --dropout=0.2 --backbone=mBERT --model_size=base
# map the result to 'w/o type' result (shown in row 9, column 4 of Table 3 and row 9, 
# column 4 of the lower part of Table 8 of the paper)
python ./XLTime/map_results.py --data_path ./XLTime-mBERT_EN_2_EU_results/


# XLTime-XLMRbase, transfer from EN to FR
# get 'w/ type' result (row 10, column 1 of the upper part of Table 8 of the paper)
python ./XLTime/main.py --data_dir_sl=./data/sl_EN_2_FR/ --data_dir_bc=./data/bc_EN_2_FR/ --output_dir=./XLTime-XLMRbase_EN_2_FR_results/ --num_train_epochs=50 --do_train --do_eval --warmup_proportion=0.5 --learning_rate=0.000007 --train_batch_size=4 --dropout=0.2 --backbone=XLMR --model_size=base
# map the result to 'w/o type' result (shown in row 10, column 1 of Table 3 and row 10, 
# column 1 of the lower part of Table 8 of the paper)
python ./XLTime/map_results.py --data_path ./XLTime-XLMRbase_EN_2_FR_results/


# XLTime-XLMRbase, transfer from EN to ES
# get 'w/ type' result (row 10, column 2 of the upper part of Table 8 of the paper)
python ./XLTime/main.py --data_dir_sl=./data/sl_EN_2_ES/ --data_dir_bc=./data/bc_EN_2_ES/ --output_dir=./XLTime-XLMRbase_EN_2_ES_results/ --num_train_epochs=100 --do_train --do_eval --warmup_proportion=0.5 --learning_rate=0.000007 --train_batch_size=4 --dropout=0.2 --backbone=XLMR --model_size=base
# map the result to 'w/o type' result (shown in row 10, column 2 of Table 3 and row 10, 
# column 2 of the lower part of Table 8 of the paper)
python ./XLTime/map_results.py --data_path ./XLTime-XLMRbase_EN_2_ES_results/


# XLTime-XLMRbase, transfer from EN to PT
# get 'w/ type' result (row 10, column 3 of the upper part of Table 8 of the paper)
python ./XLTime/main.py --data_dir_sl=./data/sl_EN_2_PT/ --data_dir_bc=./data/bc_EN_2_PT/ --output_dir=./XLTime-XLMRbase_EN_2_PT_results/ --num_train_epochs=50 --do_train --do_eval --warmup_proportion=0.5 --learning_rate=0.000007 --train_batch_size=4 --dropout=0.2 --backbone=XLMR --model_size=base
# map the result to 'w/o type' result (shown in row 10, column 3 of Table 3 and row 10, 
# column 3 of the lower part of Table 8 of the paper)
python ./XLTime/map_results.py --data_path ./XLTime-XLMRbase_EN_2_PT_results/


# XLTime-XLMRbase, transfer from EN to EU
# get 'w/ type' result (row 10, column 4 of the upper part of Table 8 of the paper)
python ./XLTime/main.py --data_dir_sl=./data/sl_EN_2_EU/ --data_dir_bc=./data/bc_EN_2_EU/ --output_dir=./XLTime-XLMRbase_EN_2_EU_results/ --num_train_epochs=50 --do_train --do_eval --warmup_proportion=0.5 --learning_rate=0.000007 --train_batch_size=4 --dropout=0.2 --backbone=XLMR --model_size=base
# map the result to 'w/o type' result (shown in row 10, column 4 of Table 3 and row 10, 
# column 4 of the lower part of Table 8 of the paper)
python ./XLTime/map_results.py --data_path ./XLTime-XLMRbase_EN_2_EU_results/


# XLTime-XLMRlarge, transfer from EN to FR
# get 'w/ type' result (row 11, column 1 of the upper part of Table 8 of the paper)
python ./XLTime/main.py --data_dir_sl=./data/sl_EN_2_FR/ --data_dir_bc=./data/bc_EN_2_FR/ --output_dir=./XLTime-XLMRlarge_EN_2_FR_results/ --num_train_epochs=50 --do_train --do_eval --warmup_proportion=0.5 --learning_rate=0.000007 --train_batch_size=4 --dropout=0.2 --backbone=XLMR --model_size=large
# map the result to 'w/o type' result (shown in row 11, column 1 of Table 3 and row 11, 
# column 1 of the lower part of Table 8 of the paper)
python ./XLTime/map_results.py --data_path ./XLTime-XLMRlarge_EN_2_FR_results/


# XLTime-XLMRlarge, transfer from EN to ES
# get 'w/ type' result (row 11, column 2 of the upper part of Table 8 of the paper)
python ./XLTime/main.py --data_dir_sl=./data/sl_EN_2_ES/ --data_dir_bc=./data/bc_EN_2_ES/ --output_dir=./XLTime-XLMRlarge_EN_2_ES_results/ --num_train_epochs=50 --do_train --do_eval --warmup_proportion=0.5 --learning_rate=0.000007 --train_batch_size=4 --dropout=0.2 --backbone=XLMR --model_size=large
# map the result to 'w/o type' result (shown in row 11, column 2 of Table 3 and row 11, 
# column 2 of the lower part of Table 8 of the paper)
python ./XLTime/map_results.py --data_path ./XLTime-XLMRlarge_EN_2_ES_results/


# XLTime-XLMRlarge, transfer from EN to PT
# get 'w/ type' result (row 11, column 3 of the upper part of Table 8 of the paper)
python ./XLTime/main.py --data_dir_sl=./data/sl_EN_2_PT/ --data_dir_bc=./data/bc_EN_2_PT/ --output_dir=./XLTime-XLMRlarge_EN_2_PT_results/ --num_train_epochs=100 --do_train --do_eval --warmup_proportion=0.5 --learning_rate=0.000007 --train_batch_size=4 --dropout=0.2 --backbone=XLMR --model_size=large
# map the result to 'w/o type' result (shown in row 11, column 3 of Table 3 and row 11, 
# column 3 of the lower part of Table 8 of the paper)
python ./XLTime/map_results.py --data_path ./XLTime-XLMRlarge_EN_2_PT_results/


# XLTime-XLMRlarge, transfer from EN to EU
# get 'w/ type' result (row 11, column 4 of the upper part of Table 8 of the paper)
python ./XLTime/main.py --data_dir_sl=./data/sl_EN_2_EU/ --data_dir_bc=./data/bc_EN_2_EU/ --output_dir=./XLTime-XLMRlarge_EN_2_EU_results/ --num_train_epochs=100 --do_train --do_eval --warmup_proportion=0.5 --learning_rate=0.000007 --train_batch_size=4 --dropout=0.2 --backbone=XLMR --model_size=large
# map the result to 'w/o type' result (shown in row 11, column 4 of Table 3 and row 11, 
# column 4 of the lower part of Table 8 of the paper)
python ./XLTime/map_results.py --data_path ./XLTime-XLMRlarge_EN_2_EU_results/


# XLTime-mBERT, transfer from EN, ES to FR
# get 'w/ type' result (row 12, column 1 of the upper part of Table 8 of the paper)
python ./XLTime/main.py --data_dir_sl=./data/sl_EN_ES_2_FR/ --data_dir_bc=./data/bc_EN_ES_2_FR/ --output_dir=./XLTime-mBERT_EN_ES_2_FR_results/ --num_train_epochs=50 --do_train --do_eval --warmup_proportion=0.5 --learning_rate=0.000007 --train_batch_size=4 --dropout=0.2 --backbone=mBERT --model_size=base
# map the result to 'w/o type' result (shown in row 12, column 1 of Table 3 and row 12, 
# column 1 of the lower part of Table 8 of the paper)
python ./XLTime/map_results.py --data_path ./XLTime-mBERT_EN_ES_2_FR_results/


# XLTime-mBERT, transfer from EN, FR to ES
# get 'w/ type' result (row 12, column 2 of the upper part of Table 8 of the paper)
python ./XLTime/main.py --data_dir_sl=./data/sl_EN_FR_2_ES/ --data_dir_bc=./data/bc_EN_FR_2_ES/ --output_dir=./XLTime-mBERT_EN_FR_2_ES_results/ --num_train_epochs=50 --do_train --do_eval --warmup_proportion=0.5 --learning_rate=0.000007 --train_batch_size=4 --dropout=0.2 --backbone=mBERT --model_size=base
# map the result to 'w/o type' result (shown in row 12, column 2 of Table 3 and row 12, 
# column 2 of the lower part of Table 8 of the paper)
python ./XLTime/map_results.py --data_path ./XLTime-mBERT_EN_FR_2_ES_results/


# XLTime-mBERT, transfer from EN, EU to PT
# get 'w/ type' result (row 12, column 3 of the upper part of Table 8 of the paper)
python ./XLTime/main.py --data_dir_sl=./data/sl_EN_EU_2_PT/ --data_dir_bc=./data/bc_EN_EU_2_PT/ --output_dir=./XLTime-mBERT_EN_EU_2_PT_results/ --num_train_epochs=50 --do_train --do_eval --warmup_proportion=0.5 --learning_rate=0.000007 --train_batch_size=4 --dropout=0.2 --backbone=mBERT --model_size=base
# map the result to 'w/o type' result (shown in row 12, column 3 of Table 3 and row 12, 
# column 3 of the lower part of Table 8 of the paper)
python ./XLTime/map_results.py --data_path ./XLTime-mBERT_EN_EU_2_PT_results/


# XLTime-mBERT, transfer from EN, FR to EU
# get 'w/ type' result (row 12, column 4 of the upper part of Table 8 of the paper)
python ./XLTime/main.py --data_dir_sl=./data/sl_EN_FR_2_EU/ --data_dir_bc=./data/bc_EN_FR_2_EU/ --output_dir=./XLTime-mBERT_EN_FR_2_EU_results/ --num_train_epochs=50 --do_train --do_eval --warmup_proportion=0.5 --learning_rate=0.000007 --train_batch_size=4 --dropout=0.2 --backbone=mBERT --model_size=base
# map the result to 'w/o type' result (shown in row 12, column 4 of Table 3 and row 12, 
# column 4 of the lower part of Table 8 of the paper)
python ./XLTime/map_results.py --data_path ./XLTime-mBERT_EN_FR_2_EU_results/


# XLTime-XLMRbase, transfer from EN, ES to FR
# get 'w/ type' result (row 13, column 1 of the upper part of Table 8 of the paper)
python ./XLTime/main.py --data_dir_sl=./data/sl_EN_ES_2_FR/ --data_dir_bc=./data/bc_EN_ES_2_FR/ --output_dir=./XLTime-XLMRbase_EN_ES_2_FR_results/ --num_train_epochs=50 --do_train --do_eval --warmup_proportion=0.5 --learning_rate=0.000007 --train_batch_size=4 --dropout=0.2 --backbone=XLMR --model_size=base
# map the result to 'w/o type' result (shown in row 13, column 1 of Table 3 and row 13, 
# column 1 of the lower part of Table 8 of the paper)
python ./XLTime/map_results.py --data_path ./XLTime-XLMRbase_EN_ES_2_FR_results/


# XLTime-XLMRbase, transfer from EN, FR to ES
# get 'w/ type' result (row 13, column 2 of the upper part of Table 8 of the paper)
python ./XLTime/main.py --data_dir_sl=./data/sl_EN_FR_2_ES/ --data_dir_bc=./data/bc_EN_FR_2_ES/ --output_dir=./XLTime-XLMRbase_EN_FR_2_ES_results/ --num_train_epochs=50 --do_train --do_eval --warmup_proportion=0.5 --learning_rate=0.000007 --train_batch_size=4 --dropout=0.2 --backbone=XLMR --model_size=base
# map the result to 'w/o type' result (shown in row 13, column 2 of Table 3 and row 13, 
# column 2 of the lower part of Table 8 of the paper)
python ./XLTime/map_results.py --data_path ./XLTime-XLMRbase_EN_FR_2_ES_results/


# XLTime-XLMRbase, transfer from EN, FR to PT
# get 'w/ type' result (row 13, column 3 of the upper part of Table 8 of the paper)
python ./XLTime/main.py --data_dir_sl=./data/sl_EN_FR_2_PT/ --data_dir_bc=./data/bc_EN_FR_2_PT/ --output_dir=./XLTime-XLMRbase_EN_FR_2_PT_results/ --num_train_epochs=50 --do_train --do_eval --warmup_proportion=0.5 --learning_rate=0.000007 --train_batch_size=4 --dropout=0.2 --backbone=XLMR --model_size=base
# map the result to 'w/o type' result (shown in row 13, column 3 of Table 3 and row 13, 
# column 3 of the lower part of Table 8 of the paper)
python ./XLTime/map_results.py --data_path ./XLTime-XLMRbase_EN_FR_2_PT_results/


# XLTime-XLMRbase, transfer from EN, FR to EU
# get 'w/ type' result (row 13, column 4 of the upper part of Table 8 of the paper)
python ./XLTime/main.py --data_dir_sl=./data/sl_EN_FR_2_EU/ --data_dir_bc=./data/bc_EN_FR_2_EU/ --output_dir=./XLTime-XLMRbase_EN_FR_2_EU_results/ --num_train_epochs=50 --do_train --do_eval --warmup_proportion=0.5 --learning_rate=0.000007 --train_batch_size=4 --dropout=0.2 --backbone=XLMR --model_size=base
# map the result to 'w/o type' result (shown in row 13, column 4 of Table 3 and row 13, 
# column 4 of the lower part of Table 8 of the paper)
python ./XLTime/map_results.py --data_path ./XLTime-XLMRbase_EN_FR_2_EU_results/


# XLTime-XLMRlarge, transfer from EN, ES to FR
# get 'w/ type' result (row 14, column 1 of the upper part of Table 8 of the paper)
python ./XLTime/main.py --data_dir_sl=./data/sl_EN_ES_2_FR/ --data_dir_bc=./data/bc_EN_ES_2_FR/ --output_dir=./XLTime-XLMRlarge_EN_ES_2_FR_results/ --num_train_epochs=50 --do_train --do_eval --warmup_proportion=0.5 --learning_rate=0.000007 --train_batch_size=4 --dropout=0.2 --backbone=XLMR --model_size=large
# map the result to 'w/o type' result (shown in row 14, column 1 of Table 3 and row 14, 
# column 1 of the lower part of Table 8 of the paper)
python ./XLTime/map_results.py --data_path ./XLTime-XLMRlarge_EN_ES_2_FR_results/


# XLTime-XLMRlarge, transfer from EN to ES
# get 'w/ type' result (row 14, column 2 of the upper part of Table 8 of the paper)
python ./XLTime/main.py --data_dir_sl=./data/sl_EN_2_ES/ --data_dir_bc=./data/bc_EN_2_ES/ --output_dir=./XLTime-XLMRlarge_EN_2_ES_results/ --num_train_epochs=50 --do_train --do_eval --warmup_proportion=0.5 --learning_rate=0.000007 --train_batch_size=4 --dropout=0.2 --backbone=XLMR --model_size=large
# map the result to 'w/o type' result (shown in row 14, column 2 of Table 3 and row 14, 
# column 2 of the lower part of Table 8 of the paper)
python ./XLTime/map_results.py --data_path ./XLTime-XLMRlarge_EN_2_ES_results/


# XLTime-XLMRlarge, transfer from EN, ES to PT
# get 'w/ type' result (row 14, column 3 of the upper part of Table 8 of the paper)
python ./XLTime/main.py --data_dir_sl=./data/sl_EN_ES_2_PT/ --data_dir_bc=./data/bc_EN_ES_2_PT/ --output_dir=./XLTime-XLMRlarge_EN_ES_2_PT_results/ --num_train_epochs=100 --do_train --do_eval --warmup_proportion=0.5 --learning_rate=0.000007 --train_batch_size=4 --dropout=0.2 --backbone=XLMR --model_size=large
# map the result to 'w/o type' result (shown in row 14, column 3 of Table 3 and row 14, 
# column 3 of the lower part of Table 8 of the paper)
python ./XLTime/map_results.py --data_path ./XLTime-XLMRlarge_EN_ES_2_PT_results/


# XLTime-XLMRlarge, transfer from EN, ES to EU
# get 'w/ type' result (row 14, column 4 of the upper part of Table 8 of the paper)
python ./XLTime/main.py --data_dir_sl=./data/sl_EN_ES_2_EU/ --data_dir_bc=./data/bc_EN_ES_2_EU/ --output_dir=./XLTime-XLMRlarge_EN_ES_2_EU_results/ --num_train_epochs=50 --do_train --do_eval --warmup_proportion=0.5 --learning_rate=0.000007 --train_batch_size=4 --dropout=0.2 --backbone=XLMR --model_size=large
# map the result to 'w/o type' result (shown in row 14, column 4 of Table 3 and row 14, 
# column 4 of the lower part of Table 8 of the paper)
python ./XLTime/map_results.py --data_path ./XLTime-XLMRlarge_EN_ES_2_EU_results/

