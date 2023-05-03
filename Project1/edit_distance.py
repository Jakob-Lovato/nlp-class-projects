### CS 691 NLP Spring 2023 Project 1
### Jakob Lovato

import numpy as np

def calc_min_edit_dist(source, target):
  n = len(source)
  m = len(target)
  
  #initialize matrix of zeros
  mat = np.array([[0 for col in range(m + 1)] for row in range(n + 1)])
  
  #initialize first row and column
  mat[:,0] = np.array(list(range(n + 1)))
  mat[0,:] = np.array(list(range(m + 1)))
  
  
  for i in range(1, n + 1):
    for j in range(1, m + 1):
      ins_cost = mat[i - 1, j] + 1
      del_cost = mat[i, j - 1] + 1
      if(source[i - 1] == target[j - 1]):
        sub_cost = mat[i - 1, j - 1]
      else:
        sub_cost = mat[i - 1, j - 1] + 2
      dist = min(ins_cost, del_cost, sub_cost)
      mat[i, j] = dist
      
  return(mat[n, m])   
  
def align(source, target):
  n = len(source)
  m = len(target)
  
  ####
  ####
  #re-implementation of calc_min_edit_dist function in order to keep array
  #initialize matrix of zeros
  mat = np.array([[0 for col in range(m + 1)] for row in range(n + 1)])
  
  #initialize first row and column
  mat[:,0] = np.array(list(range(n + 1)))
  mat[0,:] = np.array(list(range(m + 1)))
  
  
  for i in range(1, n + 1):
    for j in range(1, m + 1):
      ins_cost = mat[i - 1, j] + 1
      del_cost = mat[i, j - 1] + 1
      if(source[i - 1] == target[j - 1]):
        sub_cost = mat[i - 1, j - 1]
      else:
        sub_cost = mat[i - 1, j - 1] + 2
      dist = min(ins_cost, del_cost, sub_cost)
      mat[i, j] = dist
  ####
  ####
  
  out_source = []
  out_target = []
  operations = []

  while 1:
    current = mat[n][m]
    left = mat[n][m - 1]
    up = mat[n - 1][m]
    diag = mat[n - 1][m - 1]
    
    if source[n - 1] == target[m - 1]:
      cost = 0
    else:
      cost = 2
    
    #check if a substitution or no change happened (i.e., took diagonal path)
    if current == (diag + cost):
      out_source.append(source[n - 1])
      out_target.append(target[m - 1])
      
      if cost == 0:
        operations.append("x")
      else:
        operations.append("s")
        
      n = n - 1
      m = m - 1
      
    else:  
      #check if insertion happened (i.e., took horizontal path)
      if current == (left + 1):
        out_source.append("*")
        out_target.append(target[m - 1])
        operations.append("i")
        
        m = m - 1
        
      #check if deletion happened (i.e., took vertical path)
      elif current == (up + 1):
        out_source.append(source[n - 1])
        out_target.append("*")
        operations.append("d")
      
        n = n - 1
        
    if n == 0 and m == 0:
      out_source.reverse()
      out_target.reverse()
      operations.reverse()
      return out_source, out_target, operations
