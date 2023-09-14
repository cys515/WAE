#! /bin/bash

##EXPLAIN
# python run.py  --DataName=L1 --step=60 --model=LSTM --explain=LEMNA --segment=window --segpara=uniform
# python run.py  --DataName=L1 --step=60 --model=LSTM --explain=LIME --segment=window --segpara=uniform
# python run.py  --DataName=L1 --step=60 --model=LSTM --explain=LEMNA --segment=matrix  --segpara=slopes-not-sorted 
# python run.py  --DataName=L1 --step=60 --model=LSTM --explain=LIME --segment=matrix  --segpara=slopes-not-sorted
# python run.py  --DataName=L1 --step=60 --model=LSTM --explain=SHAP --segment=matrix  --segpara=slopes-not-sorted
# python run.py  --DataName=L1 --step=60 --model=LSTM --explain=SHAP --segment=window --segpara=uniform

# python run.py  --DataName=G1 --step=60 --model=LSTM --explain=LEMNA --segment=window --segpara=uniform
# python run.py  --DataName=G1 --step=60 --model=LSTM --explain=LIME --segment=window --segpara=uniform
# python run.py  --DataName=G1 --step=60 --model=LSTM --explain=LEMNA --segment=matrix  --segpara=slopes-not-sorted 
# python run.py  --DataName=G1 --step=60 --model=LSTM --explain=LIME --segment=matrix  --segpara=slopes-not-sorted
# python run.py  --DataName=G1 --step=60 --model=LSTM --explain=SHAP --segment=matrix  --segpara=slopes-not-sorted
# python run.py  --DataName=G1 --step=60 --model=LSTM --explain=SHAP --segment=wuniformg3indow --segpara=uniform

python run.py  --DataName=PF --step=60 --model=LSTM --explain=LEMNA --segment=window --segpara=uniform
python run.py  --DataName=PF --step=60 --model=LSTM --explain=LIME --segment=window --segpara=uniform
python run.py  --DataName=PF --step=60 --model=LSTM --explain=LEMNA --segment=matrix  --segpara=slopes-not-sorted 
python run.py  --DataName=PF --step=60 --model=LSTM --explain=LIME --segment=matrix  --segpara=slopes-not-sorted
python run.py  --DataName=PF --step=60 --model=LSTM --explain=SHAP --segment=matrix  --segpara=slopes-not-sorted
python run.py  --DataName=PF --step=60 --model=LSTM --explain=SHAP --segment=window --segpara=uniform

# python run.py  --DataName=P4 --step=60 --model=LSTM --explain=LEMNA --segment=window --segpara=uniform
# python run.py  --DataName=P4 --step=60 --model=LSTM --explain=LIME --segment=window --segpara=uniform
# python run.py  --DataName=P4 --step=60 --model=LSTM --explain=LEMNA --segment=matrix  --segpara=slopes-not-sorted 
# python run.py  --DataName=P4 --step=60 --model=LSTM --explain=LIME --segment=matrix  --segpara=slopes-not-sorted
# python run.py  --DataName=P5 --step=60 --model=LSTM --explain=SHAP --segment=matrix  --segpara=slopes-not-sorted
# python run.py  --DataName=P5 --step=60 --model=LSTM --explain=SHAP --segment=window --segpara=uniform

# python run.py  --DataName=C2 --step=60 --model=LSTM --explain=LIME --segment=matrix  --segpara=slopes-not-sorted
# python run.py  --DataName=C2 --step=60 --model=LSTM --explain=LIME --segment=window --segpara=uniform
# python run.py  --DataName=C2 --step=60 --model=LSTM --explain=LEMNA --segment=window --segpara=uniform
# python run.py  --DataName=C2 --step=60 --model=LSTM --explain=LEMNA --segment=matrix  --segpara=slopes-not-sorted
# python run.py  --DataName=C2 --step=60 --model=LSTM --explain=SHAP --segment=matrix  --segpara=slopes-not-sorted
# python run.py  --DataName=C2 --step=60 --model=LSTM --explain=SHAP --segment=window --segpara=uniform


# python run.py  --DataName=P1 --step=60 --model=LSTM --explain=LEMNA --segment=window --segpara=uniform
# python run.py  --DataName=P1 --step=60 --model=LSTM --explain=LIME --segment=window --segpara=uniform
# python run.py  --DataName=P1 --step=60 --model=LSTM --explain=LIME --segment=window --segpara=uniform
# python run.py  --DataName=P1 --step=80 --model=LSTM --explain=LIME --segment=window --segpara=uniform
# python run.py  --DataName=P1 --step=100 --model=LSTM --explain=LIME --segment=window --segpara=uniform
# python run.py  --DataName=P1 --step=120 --model=LSTM --explain=LIME --segment=window --segpara=uniform
# python run.py  --DataName=P1 --step=140 --model=LSTM --explain=LIME --segment=window --segpara=uniform
# python run.py  --DataName=P1 --step=160 --model=LSTM --explain=LIME --segment=window --segpara=uniform
# python run.py  --DataName=P1 --step=180 --model=LSTM --explain=LIME --segment=window --segpara=uniform
# python run.py  --DataName=P1 --step=200 --model=LSTM --explain=LIME --segment=window --segpara=uniform
# python run.py  --DataName=P1 --step=60 --model=LSTM --explain=LIME --segment=matrix  --segpara=slopes-not-sorted
# python run.py  --DataName=P1 --step=80 --model=LSTM --explain=LIME --segment=matrix  --segpara=slopes-not-sorted
# python run.py  --DataName=P1 --step=100 --model=LSTM --explain=LIME --segment=matrix  --segpara=slopes-not-sorted
# python run.py  --DataName=P1 --step=120 --model=LSTM --explain=LIME --segment=matrix  --segpara=slopes-not-sorted
# python run.py  --DataName=P1 --step=140 --model=LSTM --explain=LIME --segment=matrix  --segpara=slopes-not-sorted
# python run.py  --DataName=P1 --step=160 --model=LSTM --explain=LIME --segment=matrix  --segpara=slopes-not-sorted
# python run.py  --DataName=P1 --step=180 --model=LSTM --explain=LIME --segment=matrix  --segpara=slopes-not-sorted
# python run.py  --DataName=P1 --step=200 --model=LSTM --explain=LIME --segment=matrix  --segpara=slopes-not-sorted
# python run.py  --DataName=P1 --step=60 --model=LSTM --explain=SHAP --segment=matrix  --segpara=slopes-not-sorted
# python run.py  --DataName=P1 --step=60 --model=LSTM --explain=SHAP --segment=window --segpara=uniform


# python run.py  --DataName=P5 --step=60 --model=LSTM --explain=LEMNA --segment=window --segpara=uniform
# python run.py  --DataName=P5 --step=60 --model=LSTM --explain=LIME --segment=window --segpara=uniform
# python run.py  --DataName=P5 --step=60 --model=LSTM --explain=LEMNA --segment=matrix  --segpara=slopes-not-sorted
# python run.py  --DataName=P5 --step=60 --model=LSTM --explain=LIME --segment=matrix  --segpara=slopes-not-sorted
# python run.py  --DataName=P3 --step=60 --model=LSTM --explain=SHAP --segment=matrix  --segpara=slopes-not-sorted
# python run.py  --DataName=P3 --step=60 --model=LSTM --explain=SHAP --segment=window --segpara=uniform


#EVALUATION

# python evaluation.py --dataName=./Results/P5_LSTM_60_explain.dill --explain=LEMNA
# python evaluation.py --dataName=./Results/P5_LSTM_60_explain.dill --explain=LIME
# python evaluation.py --dataName=./Results/P5_LSTM_60_explain.dill --explain=SHAP

python evaluation.py --dataName=./Results/PF_LSTM_60_explain.dill --explain=LEMNA
python evaluation.py --dataName=./Results/PF_LSTM_60_explain.dill --explain=LIME
python evaluation.py --dataName=./Results/PF_LSTM_60_explain.dill --explain=SHAP

# python evaluation.py --dataName=./Results/P1_LSTM_60_explain.dill --explain=SHAP
# python evaluation.py --dataName=./Results/P1_LSTM_60_explain.dill --explain=LEMNA
# python evaluation.py --dataName=./Results/P1_LSTM_60_explain.dill --explain=LIME
# python evaluation.py --dataName=./Results/P1_LSTM_80_explain.dill --explain=LIME
# python evaluation.py --dataName=./Results/P1_LSTM_100_explain.dill --explain=LIME
# python evaluation.py --dataName=./Results/P1_LSTM_120_explain.dill --explain=LIME
# python evaluation.py --dataName=./Results/P1_LSTM_200_explain.dill --explain=LIME
# python evaluation.py --dataName=./Results/P1_LSTM_140_explain.dill --explain=LIME
# python evaluation.py --dataName=./Results/P1_LSTM_160_explain.dill --explain=LIME
# python evaluation.py --dataName=./Results/P1_LSTM_180_explain.dill --explain=LIME
# python evaluation.py --dataName=./Results/P1_LSTM_120_explain.dill --explain=LEMNA
# python evaluation.py --dataName=./Results/P1_LSTM_240_explain.dill --explain=LEMNA


# python evaluation.py --dataName=./Results/P4_LSTM_60_explain.dill --explain=LEMNA
# python evaluation.py --dataName=./Results/P4_LSTM_60_explain.dill --explain=LIME
# python evaluation.py --dataName=./Results/P4_LSTM_60_explain.dill --explain=SHAP

# python evaluation.py --dataName=./Results/L1_LSTM_60_explain.dill --explain=LEMNA
# python evaluation.py --dataName=./Results/L1_LSTM_60_explain.dill --explain=LIME
# python evaluation.py --dataName=./Results/L1_LSTM_60_explain.dill --explain=SHAP

#VISUALIZE
# python visualize.py --dataName=L1 --step=60
# python visualize.py --dataName=G1 --step=60
# python visualize.py --dataName=PF --step=60
# python visualize.py --dataName=P1 --step=60
# python visualize.py --dataName=P1 --step=80
# python visualize.py --dataName=P1 --step=100
# python visualize.py --dataName=P1 --step=120
# python visualize.py --dataName=P1 --step=140
# python visualize.py --dataName=P1 --step=160
# python visualize.py --dataName=P1 --step=180
# python visualize.py --dataName=P1 --step=200
# python visualize.py --dataName=P4 --step=60
# python visualize.py --dataName=P5 --step=60


# # #FIDELITY EVALUATION
# python fidelity.py --dataName=C1 
# python fidelity.py --dataName=G1 
# python fidelity.py --dataName=G2
# python fidelity.py --dataName=P1 
# python fidelity.py --dataName=P2
# python fidelity.py --dataName=P3 