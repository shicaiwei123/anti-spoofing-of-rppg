from PyQt5 import QtGui  # (the example applies equally well to PySide)
from PyQt5 import QtCore  # (the example applies equally well to PySide)
from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import *
from util.style import style
def getSliderEvent(slider : QSlider):
    return slider.valueChanged

def getSliderValue(slider : QSlider,r):
        t = slider.value()/100
        return (1-t) * r[0] + t * r[1] 

def create_button(layout,text,onclick):
    btn = QPushButton(text)
    btn.clicked.connect(onclick)
    layout.addWidget(btn)

def create_button_btnsettingsinclick(layout,text,settings,onclick):
    btn = QPushButton(text)
    btn.clicked.connect(lambda : onclick(btn,settings))
    layout.addWidget(btn)



def setSliderRange(slider : QSlider,range):
    slider.setMinimum(range[0])
    slider.setMaximum(range[-1])
    print(len(1/range))
    #slider.setTickInterval(1/len(range))

def setup_default_slider(layout):
    slider = QSlider(Qt.Horizontal)
    slider.setFocusPolicy(Qt.StrongFocus)
    slider.setTickPosition(QSlider.TicksBothSides)
    slider.setRange(0,100)
    slider.setSingleStep(1)
    layout.addWidget(slider)
    return slider

def create_basic_app():
    app = QtGui.QApplication([])
    app.setStyle(QStyleFactory.create('Windows'))
    app.setStyleSheet(style)
    w = QtGui.QWidget()
    p = w.palette()
    p.setColor(w.backgroundRole(), Qt.black)
    w.setPalette(p)
    return app,w

def execute_app(app,w):
    w.resize(1920,800 )
    w.show()
    app.exec_()