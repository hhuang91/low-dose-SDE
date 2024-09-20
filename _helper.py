# -*- coding: utf-8 -*-
"""
Create slice viewer for 3D volume
Use mouse wheel for scrolling through slices

@author: hhuang91
"""

import matplotlib.pyplot as plt
import torch

def sliceView(volume,vmin=None,vmax=None):
    volume = volume.squeeze().detach().cpu().numpy()  if torch.is_tensor(volume) else volume.squeeze()
    fig, ax = plt.subplots()
    ax.volume = volume
    ax.index = volume.shape[0] // 2
    ax.imshow(volume[ax.index],cmap = 'gray',
              vmin=volume.min() if vmin is None else vmin,
              vmax=volume.max() if vmax is None else vmax)
    fig.canvas.mpl_connect('scroll_event', on_scroll)
    fig.canvas.mpl_connect('key_press_event', process_key)
    
def sliceViewColor(volume):
    volume = volume.squeeze().detach().cpu().numpy()  if torch.is_tensor(volume) else volume.squeeze()
    fig, ax = plt.subplots()
    ax.volume = volume
    ax.index = volume.shape[0] // 2
    ax.imshow(volume[ax.index])
    fig.canvas.mpl_connect('scroll_event', on_scroll)
    fig.canvas.mpl_connect('key_press_event', process_key)
    
def on_scroll(event):
    fig = event.canvas.figure
    ax = fig.axes[0]
    if event.button == 'up':
        next_slice(ax)
    elif event.button == 'down':
        previous_slice(ax)
    fig.canvas.draw()

def process_key(event):
    fig = event.canvas.figure
    ax = fig.axes[0]
    if event.key == 'up':
        previous_slice(ax)
    elif event.key == 'down':
        next_slice(ax)
    fig.canvas.draw()

def previous_slice(ax):
    volume = ax.volume
    ax.index = (ax.index - 1) % volume.shape[0]  # wrap around using %
    ax.images[0].set_array(volume[ax.index])
    ax.set_xlabel('slice %s' % ax.index)

def next_slice(ax):
    volume = ax.volume
    ax.index = (ax.index + 1) % volume.shape[0]
    ax.images[0].set_array(volume[ax.index])
    ax.set_xlabel('slice %s' % ax.index)
    
def sliceCompare(vol1,vol2,vmin=None,vmax=None):
    vol1 = torch.tensor(vol1) if not torch.is_tensor(vol1) else vol1
    vol2 = torch.tensor(vol2) if not torch.is_tensor(vol2) else vol2
    volCat = torch.cat([vol1,vol2],len(vol1.shape)-1)
    sliceView(volCat,vmin,vmax)
    
def sliceCompareMany(vols):
    for vol in vols:
        vol = torch.tensor(vol) if not torch.is_tensor(vol) else vol
    volCat = torch.cat(vols,len(vol.shape)-1)
    sliceView(volCat)
