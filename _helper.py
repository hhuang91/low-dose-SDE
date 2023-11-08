# -*- coding: utf-8 -*-
"""
Create slice viewer for 3D volume
Use mouse wheel for scrolling through slices

@author: hhuang91
"""

import matplotlib.pyplot as plt
import numpy 

    
def sliceView(volume:numpy.ndarray):
    fig, ax = plt.subplots()
    ax.volume = volume
    ax.index = volume.shape[0] // 2
    ax.imshow(volume[ax.index],cmap = 'gray')
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