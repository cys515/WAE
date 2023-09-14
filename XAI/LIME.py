"""Explain a time series with LIME algorithm."""
import logging
from abc import ABC, abstractclassmethod
import numpy as np
import time

from sklearn import linear_model, metrics
from rpy2 import robjects
from rpy2.robjects.packages import importr
import rpy2.robjects.numpy2ri
from statsmodels.regression.mixed_linear_model import MixedLM
from sklearn.model_selection import train_test_split

from .perturb import Perturbation
from .segment import WindowSegmentation

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


class LimeBase(AbstractXAI):
    """Module of LIME in explaining a model."""


    def __init__(self, kernel=None, sampler=None, segmenter=None) -> None:
        """Construct perturbation base explainer.

        Args:
            kernel (obj, optional): The sklearn.linear_model for infering output of explaining model.
                Defaults to None.
            segmenter (obj, optional): Segmenation object from tsmule.sampling.segment.
                Defaults to None.
            sampler (obj, optional): Perturbation object from tsmule.sampling.perturb.
                Defaults to None.
        """
        self._kernel = kernel
        self._sampler = sampler
        self._segmenter = segmenter

        self.logger = logging.getLogger(f'::{self.__class__.__name__}::')
        self._coef = None
        self._xcoef = None


    @property
    def segment_coef(self):
        """Coefficient per segment (array).

        Returns:
            array: Array of coefficient/relevance.
        """
        return self._coef


    @property
    def coef(self):
        """Coefficient of all time points.

        Returns:
            ndarray: All coefficients of the time series. It has same shape with the time series explained.
        """
        return self._xcoef


    @staticmethod
    def _explain(samples, kernel, predict_fn):
        # Unpack samples
        new_x, z_prime, pi = list(zip(*samples)) #new_x:[n_samples,n_step,feature], z_prime:[n_samples,windows], pi:[n_samples]
        # get the predictions
        clock = time.time()
        z_hat = list(map(predict_fn, new_x))
        #z_hat=predict_fn(np.array(new_x))
        print(time.time()-clock)
        # Try to approximate g(z') ~ f(new_x) <=> g(z') = Z'* W ~ Z_hat
        _t = train_test_split(z_prime, z_hat, pi, test_size=0.3, random_state=42)  
        X, X_test, y, y_test, sw, sw_test = _t #train z_prime,test z_prime,train z_hat,test z_hat,train pi,test pi

        # Avoid nan in similarity
        sw = np.nan_to_num(np.abs(sw), 0.01)
        sw_test = np.nan_to_num(np.abs(sw_test), 0.01)

        # Fit g(z') ~ f(new_x)XZX
        if kernel == 'LIME':
            kernel_model = linear_model.Ridge(alpha=0.1)
            kernel_model.fit(X, y, sample_weight=sw)
            coef = kernel_model.coef_[0]
        elif kernel == 'LEMNA':

            r = robjects.r
            rpy2.robjects.numpy2ri.activate()
            importr('genlasso')
            importr('gsubfn')
            X_train = np.array(X)
            y_train = np.array(y)
            X_train = r.matrix(X_train, nrow = X_train.shape[0], ncol = X_train.shape[1])
            y_train = r.matrix(y_train, nrow = y_train.shape[0], ncol = y_train.shape[1])
            n = r.nrow(X_train)
            p = r.ncol(X_train)
            results = r.fusedlasso1d(y=y_train,X=X_train)
            coef = np.array(r.coef(results)[0])[:,-1]
            # X_train = np.array(X)
            # kernel_model = MixedLM(endog = y, exog = X_train, groups = sw)
            # kernel_model = kernel_model.fit()
            # coef = kernel_model.fe_params
            
        # Evaluation Score
        # y_pred = kernel_model.predict(X_test)
        # score = metrics.r2_score(y_test, y_pred)
        
        return coef


    def explain(self, x, predict_fn, segmentation_method='slopes-max', **kwargs):
        """Generate explaination for a time series.

        Args:
            x (ndarays): a time series with shape n_steps, n_features
            predict_fn (function): Function to make the prediction given the input of x.
                For keras models, the predict_fn is `keras_model.predict`.
                Because the keras model accept input of shape (n_sample, n_steps, n_features),
                so it is recommended that the model should handle both single or multiple instances.
            segmentation_method (str, optional): Segmentation method as cited in the paper.
                Defaults to 'slopes-max'.

        Returns:
            ndarray: Coefficients of all points in the time series. Same shape with the time series.
        """
        _, features = x.shape
        # Get segmentation mask, seg_m.shape == x.shape
        seg_m = self._segmenter.segment(x, segmentation_method=segmentation_method) 

        # Generate samples, sample-mask-similarity
        samples = self._sampler.perturb(x, seg_m)

        # Fitting into the model/kernel
        kernel = self._kernel
        coef= self._explain(samples, kernel, predict_fn)

        # Set coef of segments
        #xcoef = self.to_original(coef, seg_m)

        return coef,seg_m


    @staticmethod
    def to_original(coef, segments):
        """Convert coef per segment to coef per point.

        Args:
            coef (array): Coefficients of unique segments.
            segments (ndarray): Original segmentations of its time series.

        Returns:
            ndarray: coefficients of each point and have same shape with the time series (n_steps, n_features).
        """
        x_coef = np.zeros_like(segments).astype(float)

        # Get labels vectors from segmentation
        seg_unique_labels = np.unique(segments)
        assert coef.shape == seg_unique_labels.shape

        for i, l in enumerate(seg_unique_labels):
            idx = (segments == l)
            x_coef[idx] = coef[i]
        return x_coef


class LimeTS(LimeBase):
    """LIME explainer for time series."""

    def __init__(self,
                 kernel=None,
                 segmenter=None,
                 sampler=None,
                 partitions=10,
                 win_length=-1,
                 p_off=0.5,
                 replace_method='zeros',
                 n_samples=100,
                 **kwargs) -> None:
        """Construct LIME explainer for time series.

        Args:
            kernel (obj, optional): The sklearn.linear_model for infering output of explaining model.
                Defaults to None.
            segmenter (obj, optional): Segmenation object from tsmule.sampling.segment.
                Defaults to None.
            sampler (obj, optional): Perturbation object from tsmule.sampling.perturb.
                Defaults to None.
            partitions (int, optional): number of partitions.
                Defaults to 10.
            win_length (int, optional): window/subspace length.
                Defaults to -1.
            p_off (float, optional): Off probability when perturbing.
                Defaults to 0.5.
            replace_method (str, optional): Method to perturbation in tsmule.sampling.replace.
                Defaults to 'zeros'.
            n_samples (int, optional): Number of samples in perturbation.
                Defaults to 100.
        """
        kernel = kernel
        sampler = sampler or Perturbation(p_off, replace_method, n_samples)
        segmenter = segmenter or WindowSegmentation(partitions, win_length)

        super().__init__(kernel=kernel, sampler=sampler, segmenter=segmenter)
