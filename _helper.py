# -*- coding: utf-8 -*-
"""
Created on Wed Mar 16 10:57:33 2022

@author: hhuang91
"""

import matplotlib.pyplot as plt

# class IndexTracker:
#     def __init__(self, ax, X):
#         self.ax = ax
#         ax.set_title('use scroll wheel to navigate images')

#         self.X = X
#         rows, cols, self.slices = X.shape
#         self.ind = self.slices//2

#         self.im = ax.imshow(self.X[:, :, self.ind],cmap='gray')
#         # self.update()

#     def on_scroll(self, event):
#         print("%s %s" % (event.button, event.step))
#         if event.button == 'up':
#             self.ind = (self.ind + 1) % self.slices
#         else:
#             self.ind = (self.ind - 1) % self.slices
#         self.update()
    
#     def keyPress(self, event):
#         if event.key == 'z':
#             self.ind = (self.ind + 1) % self.slices
#         elif event.key == 'x':
#             self.ind = (self.ind - 1) % self.slices
#         self.update(event)
        
#     def update(self,event):
#         fig = event.canvas.figure
#         ax = fig.axes[0]
#         ax.images[0].set_array(self.X[self.ind])
#         fig.canvas.draw()
#         # self.im.set_data(self.X[:, :, self.ind])
#         # self.ax.set_ylabel('slice %s' % self.ind)
#         # self.im.axes.figure.canvas.draw()

# def sliceViewer(X):
#     fig, ax = plt.subplots(1, 1)
#     tracker = IndexTracker(ax, X)
#     fig.canvas.mpl_connect('scroll_event', tracker.on_scroll)
#     fig.canvas.mpl_connect('key_press_event', tracker.keyPress)
#     plt.show()
    
def sliceView(volume):
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