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
    self.sto_grad_ascent()
    inde_comp = np.dot(self.unmix_mat, self.data.T)
    return (inde_comp.T)

  # apply gradient ascent to update unmixing matrix
  # maximum likelihood version ICA is implemented
  def sto_grad_ascent(self):
    step_len = 0.1
    num_iteration = 10000
    for index in range(1, num_iteration):
      for index1 in range(0, self.data.shape[0]):
        # create vector for [1-2g(w1.x),...,1-2g(wn.x)]
        vec = self.create_vec(self.data[index1, :])
        inv_mat = LA.inv(self.unmix_mat.T)
        outer_prod = np.outer(vec, self.data[index1, :])
        # update unmixing matrix
        self.unmix_mat += (step_len * (outer_prod + inv_mat))
        self.unmix_mat = self.row_norm(self.unmix_mat)

  def create_vec(self, data_point):
    dim = self.unmix_mat.shape[0]
    vec = []
    for index in range(0, dim):
      temp = np.dot(self.unmix_mat[index, :], data_point)
      vec = np.append(vec, 1 - 2 * self.sigmoid(temp))
    return vec

  def sigmoid(self, x):
    return (1 / (1 + math.exp(-x)))

  def row_norm(self, unmix_mat):
    for index in range(0, unmix_mat.shape[0]):
      row_vec = unmix_mat[index, :]
      magnitude = LA.norm(row_vec)
      row_vec /= magnitude
      unmix_mat[index,:] = row_vec
    return unmix_mat
