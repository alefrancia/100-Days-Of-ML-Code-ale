# Diamonds price prediction based on their cut, color, clarity, price with PyTorch
# https://github.com/vineel369/DeepLearning-with-PyTorch/blob/master/01_Linear%20Regression/Linear_Regression_with_Pytorch_Diamond_Price_prediction_final.ipynb
# SIN NVIDA
# Conda
# conda install pytorch torchvision cpuonly -c pytorch

# Pip
# pip install torch==1.5.0+cpu torchvision==0.6.0+cpu -f https://download.pytorch.org/whl/torch_stable.html

from __future__ import print_function, division

# Imports
import torch
import numpy as np
import jovian
import torchvision
import torch.nn as nn
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import torch.nn.functional as F
from torchvision.datasets.utils import download_url
from torch.utils.data import DataLoader, TensorDataset, random_split

import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

# plt.ion()   # interactive mode

data = pd.read_csv('datasets\diamonds.csv')
data.head()
print(data.head())

data.drop(columns = 'Unnamed: 0', axis = 1, inplace = True )
print(data.head())

data.shape
data.info()
data.describe()
temp = data[['x','y','z']].replace(0,np.NaN)
temp.isnull().sum()

data = data.loc[(data[['x','y','z']]!=0).all(axis=1)]
data.shape

data.describe()


data.isnull().any()

# data.hist(bins=50,figsize=(20,15))

# sns.pairplot(data , diag_kind = 'kde');


plt.figure(figsize = (10,5))
sns.heatmap(data.corr(),annot = True , cmap = 'coolwarm' );

corr_mat = data.corr()
plt.figure(figsize = (10,5))
corr_mat['price'].sort_values(ascending = False).plot(kind = 'bar');

input_cat_columns = data.select_dtypes(include = ['object']).columns.tolist()

for col in input_cat_columns:
    sns.catplot(x=col, y="price",
            kind="box", dodge=False, height = 5, aspect = 3,data=data);

data_one_hot_encoding = pd.get_dummies(data)
data_one_hot_encoding.head()

print(data_one_hot_encoding.columns.values)
plt.show()

