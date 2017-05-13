import numpy as np
from numpy import linalg as LA
import math
class ICA:
  # creating data members accessible to all objects
  # ...

  # creating object private data members
  def __init__(self, data_matrix):
    self.data = data_matrix
    # find mean of each component
    self.mean = self.find_mean()
    # find standard deviation of each component
    self.std = self.find_std()
    # now each component has 0 mean 1 variance
    self.norm_data = self.normalize()
    # define unmixing matrix
    n = self.data.shape[1]
    self.unmix_mat = np.random.uniform(0,1,(n,n))
    
  # find mean of data points along each column
  def find_mean(self):
    return np.mean(self.data)

  # find mean of data points along each column
  def find_std(self):
    return np.std(self.data)

  # normalize data points along each component
  def normalize(self):
    return ((self.data - self.mean) / self.std)

  # find source components from observation components
  def find_sources(self):
    self.find_unmix_mat()
    inde_comp = np.dot(self.unmix_mat, self.data.T)
    return (inde_comp.T)

  # finding unmixing matrix by calculating each row
  def find_unmix_mat(self):
    # iterate for each row of unmixing_matrix
    for row_index in range(0, self.data.shape[1]):
      self.unmix_mat[row_index] = \
          self.fastica_one_comp(unmix_mat[row_index])
