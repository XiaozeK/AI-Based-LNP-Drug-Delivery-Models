from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from datetime import datetime

from scipy.stats import uniform, randint

from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler, RobustScaler, MinMaxScaler

from sklearn.model_selection import cross_val_score, GridSearchCV, KFold, RandomizedSearchCV, train_test_split

from tsfresh import extract_features, extract_relevant_features, select_features
from tsfresh.utilities.dataframe_functions import impute
from tsfresh.feature_extraction import ComprehensiveFCParameters, EfficientFCParameters, MinimalFCParameters, settings

# set the logger to error level
# tsfresh outputs many warnings for features that cannot be calculated
import logging
logging.basicConfig(level=logging.ERROR)

import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

# from tsfresh github examples
# note in this function they write normalize where they should really have written standardize!
class PCAForPandas(PCA):
    """This class is just a small wrapper around the PCA estimator of sklearn 
    including normalization to make it compatible with pandas DataFrames."""

    def __init__(self, 