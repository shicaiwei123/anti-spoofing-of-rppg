import pyqtgraph as pg
import numpy as np
import matplotlib.cm as cm
def getcolors(ncolors):
    colors= cm.rainbow(np.linspace(0, 1, ncolors))    
    return colors[:,:3]*255




def plot(fig,x,y,color):
    plot = fig.plot(x,y,pen = (color[0],color[1],color[2]))
    return plot

def create_fig():
    fig = pg.PlotWidget()
    fig.showGrid(x=True, y=True)
    return fig


def addLabels(fig,xlabel,ylabel,xunit ='-',yunit = '-'):
    fig.setLabel('left', ylabel, units=xunit)
    fig.setLabel('bottom', xlabel, units=yunit)
