#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug 25 16:22:43 2021

@author: ning
"""

import torch

from matplotlib import pyplot as plt

n_points = 1000
n_gaps = 250
loss_function = torch.nn.BCELoss()
predictions = torch.linspace(0,1,n_points)
ground_true = torch.ones(n_points) * 0.5

losses = [loss_function(y_pred,y_true).item() for y_pred,y_true in zip(predictions,ground_true)]
x = predictions.numpy()

fig,axes = plt.subplots(figsize = (10,5),
                        ncols = 2,
                        sharey = False,
                        sharex = False,
                        )
ax = axes[0]
ax.plot(x,losses)
ax.set(xlabel = 'Prediction',
       ylabel = 'Loss')
ax = axes[1]
ax.plot(x[1:-1],losses[1:-1])
ax.set(xlabel = 'Prediction',
       ylabel = 'Loss')