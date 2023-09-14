from re import A
from tkinter import YES
import pandas as pd 
from tensorflow import keras
import matplotlib.pyplot as plt
from matplotlib.colorbar import ColorbarBase as cb
import numpy as np
import dill
from math import log
from sympy import Symbol,solveset,nsolve,S,I,nsimplify,solve,Add,re
from sympy.solvers.solveset import _transolve as transolve
from sympy.solvers.solveset import _solve_logarithm as solve_log
from sklearn.preprocessing import MinMaxScaler
from XAI.segment import WindowSegmentation
from XAI.LIME import LimeTS
from XAI.segment import SegmentationPicker
from torch.utils.data import DataLoader, TensorDataset
from tensorflow import keras
import torch
import dill 
from captum.attr import Saliency,IntegratedGradients
# ImageClassifier takes a single input tensor of images Nx3x32x32,
# and returns an Nx10 tensor of class probabilities.
# Generating random input with size 2x3x32x32

with open("./Results-len20/L1_LSTM_60_explain.dill", 'rb') as f:
    dataset = dill.load(f)
print("L1_LIME_win_uni:",sum(dataset["res_LIME_window_uniform"])/len(dataset["res_LIME_window_uniform"]))
print("L1_LEMNA_win_uni:",sum(dataset["res_LEMNA_window_uniform"])/len(dataset["res_LEMNA_window_uniform"]))
print("L1_LIME_matrix:",sum(dataset["res_LIME_matrix_slopes-not-sorted"])/len(dataset["res_LIME_matrix_slopes-not-sorted"]))
print("L1_LEMNA_matrix:",sum(dataset["res_LEMNA_matrix_slopes-not-sorted"])/len(dataset["res_LEMNA_matrix_slopes-not-sorted"]))
print("L1_SHAP_uni:",sum(dataset["res_SHAP_window_uniform"])/len(dataset["res_SHAP_window_uniform"]))
print("L1_SHAP_matrix:",sum(dataset["res_SHAP_matrix_slopes-not-sorted"])/len(dataset["res_SHAP_matrix_slopes-not-sorted"]))
with open("./Results-len20/P1_LSTM_60_explain.dill", 'rb') as f:
    dataset1 = dill.load(f)
print("P1_LIME_win_uni:",sum(dataset1["res_LIME_window_uniform"])/len(dataset1["res_LIME_window_uniform"]))
print("P1_LEMNA_win_uni:",sum(dataset1["res_LEMNA_window_uniform"])/len(dataset1["res_LEMNA_window_uniform"]))
print("P1_LIME_matrix:",sum(dataset1["res_LIME_matrix_slopes-not-sorted"])/len(dataset1["res_LIME_matrix_slopes-not-sorted"]))
print("P1_LEMNA_matrix:",sum(dataset1["res_LEMNA_matrix_slopes-not-sorted"])/len(dataset1["res_LEMNA_matrix_slopes-not-sorted"]))
print("P1_SHAP_uni:",sum(dataset1["res_SHAP_window_uniform"])/len(dataset1["res_SHAP_window_uniform"]))
print("P1_SHAP_matrix:",sum(dataset1["res_SHAP_matrix_slopes-not-sorted"])/len(dataset1["res_SHAP_matrix_slopes-not-sorted"]))
with open("./Results-len20/P5_LSTM_60_explain.dill", 'rb') as f:
    dataset2 = dill.load(f)
print("P4_LIME_win_uni:",sum(dataset2["res_LIME_window_uniform"])/len(dataset2["res_LIME_window_uniform"]))
print("P4_LEMNA_win_uni:",sum(dataset2["res_LEMNA_window_uniform"])/len(dataset2["res_LEMNA_window_uniform"]))
print("P4_LIME_matrix:",sum(dataset2["res_LIME_matrix_slopes-not-sorted"])/len(dataset2["res_LIME_matrix_slopes-not-sorted"]))
print("P4_LEMNA_matrix:",sum(dataset2["res_LEMNA_matrix_slopes-not-sorted"])/len(dataset2["res_LEMNA_matrix_slopes-not-sorted"]))
print("P4_SHAP_uni:",sum(dataset2["res_SHAP_window_uniform"])/len(dataset2["res_SHAP_window_uniform"]))
print("P4_SHAP_matrix:",sum(dataset2["res_SHAP_matrix_slopes-not-sorted"])/len(dataset2["res_SHAP_matrix_slopes-not-sorted"]))
with open("./Results/PF_LSTM_60_explain.dill", 'rb') as f:
    dataset3 = dill.load(f)
print("PF_LIME_win_uni:",sum(dataset3["res_LIME_window_uniform"])/len(dataset3["res_LIME_window_uniform"]))
print("PF_LEMNA_win_uni:",sum(dataset3["res_LEMNA_window_uniform"])/len(dataset3["res_LEMNA_window_uniform"]))
print("PF_LIME_matrix:",sum(dataset3["res_LIME_matrix_slopes-not-sorted"])/len(dataset3["res_LIME_matrix_slopes-not-sorted"]))
print("PF_LEMNA_matrix:",sum(dataset3["res_LEMNA_matrix_slopes-not-sorted"])/len(dataset3["res_LEMNA_matrix_slopes-not-sorted"]))
print("PF_SHAP_uni:",sum(dataset3["res_SHAP_window_uniform"])/len(dataset3["res_SHAP_window_uniform"]))
print("PF_SHAP_matrix:",sum(dataset3["res_SHAP_matrix_slopes-not-sorted"])/len(dataset3["res_SHAP_matrix_slopes-not-sorted"]))
print("11")

# Datafile = "./G1120.dill"
# model_name="./Models/G1_LSTM120"
# device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
# model = torch.load(model_name).to(device)
# input = dataset["samples"][0:1]
# input = torch.from_numpy(input).float().to(device)
# # Defining Saliency interpreter
# saliency = Saliency(model)
# # Computes saliency maps for class 3.
# attribution = saliency.attribute(input)
# attr = attribution.cpu().detach().numpy().ravel()
# print(attr.shape())
# print(attr)

# with open(Datafile, 'rb') as f:
#     dataset = dill.load(f)
# trainX,trainY,testX,testY= dataset[0],dataset[1],dataset[2],dataset[3]
# train_dataset = TensorDataset(torch.Tensor(trainX), torch.Tensor(trainY))
# test_dataset = TensorDataset(torch.Tensor(testX), torch.Tensor(testY))
# train_loader = DataLoader(dataset=train_dataset ,batch_size=300,shuffle=False,num_workers=0)
# test_loader = DataLoader(dataset=test_dataset,batch_size=300,shuffle=False,num_workers=0)
# Mean_input = next(iter(test_loader))[0][1:].to(device)
# ig = IntegratedGradients(model,multiply_by_inputs=True)

# def Base_mean(DF):
#     COPY = DF.copy()
#     for i in np.arange(298,93,-1):
#         COPY.iloc[i,:] = np.mean(np.array(Test_df2.iloc[i-92:i-1,:]),axis = 0)
#     return COPY   
# class Sequential_Data(Dataset):
#     def __init__(self, data, window):
#         self.data = torch.Tensor(data.values)
#         self.window = window
#         self.shape = self.__getshape__()
#         self.size = self.__getsize__()
 
#     def __getitem__(self, index):
#         x = self.data[index:index+self.window]
#         y = self.data[index+6+self.window,3]
#         return x, y
 
#     def __len__(self):
#         return len(self.data) -  self.window -7
    
#     def __getshape__(self):
#         return (self.__len__(), *self.__getitem__(0)[0].shape)
    
#     def __getsize__(self):
#         return (self.__len__())


# Mean_df = Base_mean(Test_df2)
# Mean_data = TensorDataset(dataset=Mean_df)
# Mean_loader = DataLoader(dataset=Mean_data,batch_size=len(Mean_data),shuffle=False,num_workers=0)
# Mean_baseline = next(iter(Mean_loader))[0]
# Mean_baseline = Mean_baseline[1:].to(device)
# at_mean, mean_error = ig.attribute(Mean_input,
#                                          baselines=Mean_baseline,
#                                          method='gausslegendre',
#                                          return_convergence_delta=True,
#                                          n_steps = 18)

# at_mean = np.mean(np.array(at_mean.cpu()),axis=0)

