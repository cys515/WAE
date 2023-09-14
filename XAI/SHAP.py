import math
import logging
import torch
from abc import ABC, abstractclassmethod
import numpy as np
import shap
import time
from copy import deepcopy
import warnings
import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128"
from .segment import WindowSegmentation
warnings.filterwarnings('ignore')

class AbstractXAI(ABC):
	"""Abstract module for explainable AI."""


	@abstractclassmethod
	def __init__(self) -> None:
		"""Abstract construct."""
		pass


	@abstractclassmethod
	def explain(self, x, predict_fn, **kwargs):
		"""Generate explaination for time series x, given a model or predict function.

		Args:
			x (ndarray): Time series x with (n_steps, n_features)
			predict_fn (function): Predict function of the model.
		"""
		pass


class ShapBase(AbstractXAI):
	def __init__(self, kernel, segmenter, model, background_data, ts_data) -> None:
		"""Construct perturbation base explainer.

		Args:
			kernel (obj): The sklearn.linear_model for infering output of explaining model.
				Defaults to None.
			model (obj): A model object that will be used for prediction. The model object must have a method called predict() which produces the model output for a given input.
			segmenter (obj): Segmenation object from tsmule.sampling.segment.
				Defaults to None.
		"""
		self._kernel = kernel
		self._segmenter = segmenter
		self.background_data = background_data
		self.model = model
		self.ts_data = ts_data
		self.seg_m = None

		self.logger = logging.getLogger(f'::{self.__class__.__name__}::')
		self._coef = None
		self._xcoef = None

		self.num_ts_ftr = background_data.shape[2]
		self.num_ts_step = background_data.shape[1]

		# Creating all data (background and test together)
		self.all_ts = np.concatenate((self.background_data, ts_data[np.newaxis,:]), axis=0)

	def wraper_predict(self, x):
		assert len(x.shape) == 2

		ts_x = x[:, 0:].copy()
		tmp_seg = np.unique(self.seg_m)
		# initializing the value of all arrays
		ts_x_ = np.zeros((x.shape[0], self.all_ts.shape[1], self.all_ts.shape[2]))

		for i in range(x.shape[0]):
			# creating time series data
			for t in range(self.num_ts_step):
				for j in range(self.num_ts_ftr):
					# Finding the corresponding time interval
					wind_t = np.where(tmp_seg==self.seg_m[t,j])
					ind = ts_x[i, wind_t]
					ts_x_[i, t, j] = self.all_ts[ind, t, j]
					
		if hasattr(torch.cuda, 'empty_cache'):
			torch.cuda.empty_cache()
		device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
		ts_x_ = torch.from_numpy(ts_x_).float().to(device)
		tmp = self.model(ts_x_)

		return tmp.cpu().detach().numpy().ravel()
	
	def explain(self, segmentation_method='slopes-max', **kwargs):
		"""Generate explaination for a time series.

		Args:
			x (ndarays): a time series with shape n_steps, n_features
			segmentation_method (str, optional): Segmentation method as cited in the paper.
				Defaults to 'slopes-max'.

		Returns:
			ndarray: Coefficients of all points in the time series. Same shape with the time series.
		"""

		# Get segmentation mask, seg_m.shape == x.shape
		self.seg_m = self._segmenter.segment(self.ts_data, segmentation_method=segmentation_method) 
		
		total_num_features = len(np.unique(self.seg_m))
		bg_data = np.array([[i]*total_num_features for i in range(0,self.background_data.shape[0])])
		test_data = np.array([[i]*total_num_features for i in range(self.background_data.shape[0],self.background_data.shape[0]+1)])

		if self._kernel == 'Kernel':
			self.explainer = shap.KernelExplainer(self.wraper_predict, bg_data)


		shap_values = self.explainer.shap_values(test_data)
		shap_values = np.array(shap_values)
		min = shap_values.min()
		if min<0:
			shap_values = shap_values - min 
		return shap_values, self.seg_m
	
class ShapTS(ShapBase):
	"""SHAP explainer for time series."""

	def __init__(self,
				 kernel='Kernel',
				 model = None,
				 segmenter=None,
				 partitions=10,
				 win_length=-1,
				 background_data = None,
				 ts_data= None,
				 **kwargs) -> None:
		"""Construct LIME explainer for time series.

		Args:
			kernel (str, optional): The shap.Explainer for infering output of explaining model.
				Defaults to Kernel.
			segmenter (obj, optional): Segmenation object from tsmule.sampling.segment.
				Defaults to None.
			partitions (int, optional): number of partitions.
				Defaults to 10.
			win_length (int, optional): window/subspace length.
				Defaults to -1.
			background_data (obj, optional): background data.
				Defaults to None.
		"""
		kernel = kernel
		segmenter = segmenter or WindowSegmentation(partitions, win_length)

		super().__init__(kernel=kernel, segmenter=segmenter, background_data = background_data, model = model, ts_data=ts_data)
