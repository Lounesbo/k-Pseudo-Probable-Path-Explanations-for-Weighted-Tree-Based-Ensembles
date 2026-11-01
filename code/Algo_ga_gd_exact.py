import numpy as np
import pandas as pd
from tqdm import tqdm
import time
import sys
import os
import operator
import json
import functools
import random
import copy
import uuid
import subprocess
import itertools
import math

from itertools import combinations
from random import shuffle
from collections import OrderedDict
from multiprocessing import freeze_support

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.ticker as ticker

import my_tree as mt
import my_forest as mf

sys.path.append('build')

from sklearn.model_selection import train_test_split, LeaveOneGroupOut
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn import tree

from pysat.formula import WCNF, CNF
from pysat.pb import *
from pysat.card import *
from pysat.examples.rc2 import RC2
from pysat.solvers import Solver, Glucose4, Minisat22

def M_s(phi, t, n):
    t_set = set(t)
    t_neg = {-c for c in t}
    w = 0
    for cl in phi:
        cl_set = set(cl)
        if cl_set & t_neg:
            continue
        union_size = len(cl_set | t_set)
        w += 1 << (n - union_size)
    return w

def W_s(phi, t, n):
    return M_s(phi, t, n) / (1 << (n - len(t)))

def mu_forest(S, Phi, weights, n):
    t = list(S)
    return sum(w * M_s(phi, t, n) for phi, w in zip(Phi, weights))

def epsilon_forest(S, Phi, weights, n):
    sum_weights = sum(weights)
    return mu_forest(S, Phi, weights, n) / (sum_weights * (1 << (n - len(S))))

def exact_search(k, I, Phi, weights, n):
    best_S, best_eps = None, float('inf')
    for size in range(1, k+1):
        for S in combinations(I, size):
            eps = epsilon_forest(list(S), Phi, weights, n)
            if eps < best_eps:
                best_eps, best_S = eps, list(S)
                if best_eps == 0:  # on ne peut pas faire mieux
                    return best_S, best_eps
    return best_S, best_eps

def greedy_ascent(k, I, Phi, weights, n, c_f=1):
    
    S = []
    history = [S.copy()]
    
    mu_empty = mu_forest([], Phi, weights, n)
    gamma = max(1/math.e, c_f)
    
    while len(S) < k:
        best_i, best_val = None, float('inf')
        for i in I:
            if i not in S:
                val = mu_forest(S + [i], Phi, weights, n)
                if val < best_val:
                    best_val, best_i = val, i
        
        S = S + [best_i]
        history.append(S.copy())
        
        mu_S = mu_forest(S, Phi, weights, n)
        if mu_S == 0:
            break
        limit = k * math.ceil(math.log(mu_empty / (gamma * mu_S)))
        if len(S) >= limit:
            break
    
    return min(history, key=lambda s: epsilon_forest(s, Phi, weights, n))

def greedy_descent(k, I, Phi, weights, n):
    S = I.copy()
    history = [S.copy()]
    
    while len(S) > 0:
        best_i, best_val = None, float('inf')
        for i in S:
            val = mu_forest([x for x in S if x != i], Phi, weights, n)
            if val < best_val:
                best_val, best_i = val, i
        
        S = [x for x in S if x != best_i]
        history.append(S.copy())
    
    # retourner le meilleur parmi les solutions de taille ≤ k
    candidates = [s for s in history if len(s) <= k]
    return min(candidates, key=lambda s: epsilon_forest(s, Phi, weights, n))

def curvature(I, Phi, weights, n):
    t_I = list(I)
    mu_I = mu_forest(t_I, Phi, weights, n)
    ratios = []
    for i in I:
        Li_full   = mu_forest([x for x in t_I if x != i], Phi, weights, n) - mu_I
        Li_single = mu_forest([], Phi, weights, n) - mu_forest([i], Phi, weights, n)
        if Li_single > 0:
            ratios.append(Li_full / Li_single)
    return 1.0 - min(ratios) if ratios else 1.0
