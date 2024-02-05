"""
This utils script provides some necessary/popular functions used in our PFR frameworks
"""
import numpy as np, pandas as pd
from scipy.stats import norm
import copy, os


def is_pos_def(Sigma):
  """
  Checking if matrix X is PSD
  :param x: dx1 covariance matrix
  :return:
  """
  return np.all(np.linalg.eigvals(Sigma) > 0)

def get_entropy(p):
  """
  Compute the entropy of the prediction
  p is a softmax probability vector/list. p_i = p(Y = i|x)
  Sum of entries in p should be equal to 1.
  Args:
    p : numpy Cx1  here we have C classes
  Return:
    a single number which is entropy of distribution p
  """
  p_clip = np.clip(p, 1e-8, 1 - 1e-8)
  p_clip = p_clip/np.sum(p_clip)

  return np.sum( [-x * np.log(x) for x in p_clip])

def powerset(s):
  """
  Return all possible subset of s. The number of subsets is 2^|s|.
  """
  x = len(s)
  subset = []
  for i in range(1 << x):
    subset.append([s[j] for j in range(x) if (i & (1 << j))])

  return subset

def linspace_md(v_min, v_max, dim, num):
  """
  Generate all equally spaced samples in a cubic of dim dimensional
  The total number of samples return must be num^dim
  When dim =1, then this function resembles np.linspace(v_min, v_max, num)
  In our paper, v_min =-1, v_max =1 which is feature range.
  """
  output = np.empty((num ** dim, dim))
  values = np.linspace(v_min, v_max, num)
  for i in range(output.shape[0]):
    for d in range(dim):
      output[i][d] = values[(i // (dim ** d)) % num]
  return output



