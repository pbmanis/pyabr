#!/usr/bin/env python


import sys
import datetime
import pickle
import platform

import numpy as np
import scipy.signal
import pyqtgraph as pg

# from PyQt4 import QtGui, QtCore
from pyqtgraph.parametertree import Parameter, ParameterTree

import PySounds


class PyABR(object):
    def __init__(self):

        #    ecg = MeasureECG(knownFiles[fname])
        # check hardware first
        self.PS = PySounds.PySounds()
        self.hw, self.sfin, self.sfout = self.PS.getHardware()
        self.stimpars = {
            "Stim_type": "click",
            "Dur": 1.0,
            "Per": 25.0,
            "Reps": 500,
            "Pip_dur": 2.5,
            "Pip_rf": 0.5,
            "click_duration": 0.1,
            "click_polarity": "+",
            "click_alternate": True,
            "LPF": 3000.0,
            "Filename": "test.abr",
            "Info": "Subject Info",
        }
        print(f"HW: ", self.hw)
        self.buildGUI()

    def buildGUI(self):
        # Build GUI and window

        app = pg.mkQApp()
        win = pg.QtGui.QWidget()
        layout = pg.QtGui.QGridLayout()
        win.setLayout(layout)
        win.setWindowTitle("ABR Acquistion")
        win.resize(1024, 800)

        # Define parameters that control aquisition and buttons...
        params = [
            {
                "name": "Acquisition Parameters",
                "type": "group",
                "children": [
                    {
                        "name": "ABR Recording Duration",
                        "type": "float",
                        "value": self.stimpars["Dur"],  # in msec
                        "limits": [1.0, 2000.0],
                        "suffix": "ms",
                        "default": 10.0,
                    },
                    {
                        "name": "Stimulus Period",
                        "type": "float",
                        "value": self.stimpars["Per"],  # msec
                        "limits": [10.0, 1000.0],
                        "suffix": "ms",
                        "default": 25.0,
                    },
                    {
                        "name": "Repetitions",
                        "type": "int",
                        "value": self.stimpars["Reps"],  # msec
                        "limits": [1, 2000],
                        "suffix": "",
                        "default": 25.0,
                    },
                    {
                        "name": "Stim Type",
                        "type": "str",
                        "value": self.stimpars["Stim_type"],
                        "values": ["tonepip", "click", "noisepip"],
                        "default": "click",
                    },
                    {
                        "name": "Pip Duration",
                        "type": "float",
                        "value": self.stimpars["Pip_dur"],
                        "step": 0.1,
                        "limits": [0.5, 10.0],
                        "suffix": "ms",
                        "default": 1.0,
                    },
                    {
                        "name": "Pip RF",
                        "type": "float",
                        "value": self.stimpars["Pip_rf"],
                        "step": 0.1,
                        "limits": [0.1, 10.0],
                        "suffix": "ms",
                        "default": 0.5,
                    },
                    {
                        "name": "Click Duration",
                        "type": "float",
                        "value": self.stimpars["click_duration"],
                        "step": 0.01,
                        "limits": [0.02, 1.0],
                        "suffix": "ms",
                        "default": 0.1,
                    },
                    {
                        "name": "Click Polarity",
                        "type": "str",
                        "values": ["+", "-"],
                        "value": self.stimpars["click_polarity"],
                        "default": "+",
                    },
                    {
                        "name": "Alternate Polarity",
                        "type": "bool",
                        "value": self.stimpars["click_alternate"],
                        "default": True,
                    },
                    {
                        "name": "LPF",
                        "type": "float",
                        "value": self.stimpars["LPF"],
                        "step": 5.0,
                        "limits": [300.0, 3000.0],
                        "suffix": "Hz",
                        "default": 3000.0,
                    },
                    {
                        "name": "Filename",
                        "type": "str",
                        "value": self.stimpars["Filename"],
                        "default": "test.abr",
                    },
                    {"name": "Info", "type": "text", "value": self.stimpars["Info"],},
                    #        ]},
                    #    {'name': 'Actions', 'type': 'group', 'chidren': [
                    {"name": "New Filename", "type": "action"},
                    {"name": "Start New", "type": "action"},
                    {"name": "Stop/Pause", "type": "action"},
                    {"name": "Continue", "type": "action"},
                    {"name": "Save Visible", "type": "action"},
                    {"name": "Load File", "type": "action"},
                ],
            },
        ]
        ptree = ParameterTree()
        self.ptreedata = Parameter.create(name="params", type="group", children=params)
        ptree.setParameters(self.ptreedata)

        # build layout for plots and parameters
        layout.addWidget(ptree, 0, 0, 5, 2)  # Parameter Tree on left

        # add space for the graphs
        view = pg.GraphicsView()
        l = pg.GraphicsLayout(border=(50, 50, 50))
        view.setCentralItem(l)
        layout.addWidget(view, 0, 3, 5, 3)  # data plots on right

        plt_hr = l.addPlot()
        plt_hr.getAxis("left").setLabel("ABR Signal", color="#ff0000")
        plt_hr.setTitle("uV", color="#ff0000")
        plt_hr.getAxis("bottom").setLabel("t (msec)", color="#ff0000")
        plt_hr.setYRange(-5.0, 5.0)
        ## create a new ViewBox, link the right axis to its coordinate system
        l.nextRow()
        plt_var = l.addPlot()  #
        # pg.ViewBox(parent=plt_hr) # trying to put on one axis, but doesn't work following example
        # plt_hr.showAxis('right')
        # plt_hr.scene().addItem(plt_var)  # add variance to HR plot scene
        # plt_hr.getAxis('right').linkToView(plt_var)  # add view of Y axes
        plt_var.setXLink(plt_hr)
        # plt_hr.getAxis('right').setLabel('HR Variability', color='#0000ff')
        plt_var.getAxis("left").setLabel("V", color="#0000ff")
        plt_var.setTitle("Stimulus", color="#ff0000")
        plt_var.setYRange(0, 20)
        plt_var.getAxis("bottom").setLabel("t (msec)", color="#0000ff")

        l.nextRow()
        l2 = l.addLayout(colspan=3, border=(50, 0, 0))  # embed a new layout
        l2.setContentsMargins(10, 10, 10, 10)
        plt_first = l2.addPlot(Title="Amplitude-Intensity")
        plt_first.getAxis("bottom").setLabel("t (s)")
        plt_first.getAxis("left").setLabel("V")
        plt_first.setTitle("First Template")
        self.waveform = l2.addPlot(Title="Waveforms")
        self.waveform.getAxis("bottom").setLabel("t (s)")
        self.waveform.getAxis("left").setLabel("V")
        self.waveform.setTitle("Current")
        plt_RRI = l2.addPlot(Title="RRI")
        plt_RRI.setTitle("RRI")
        plt_RRI.getAxis("bottom").setLabel("t (s)")
        plt_RRI.getAxis("left").setLabel("t (s)")

        win.show()
        self.ptreedata.sigTreeStateChanged.connect(self.command_dispatcher)

        if (sys.flags.interactive != 1) or not hasattr(pg.QtCore, "PYQT_VERSION"):
            pg.QtGui.QApplication.instance().exec_()

    def command_dispatcher(self, param, changes):

        for param, change, data in changes:
            path = self.ptreedata.childPath(param)
            print("Path: ", path[1])
            if path[1] == "Start New":
                self.make_waveforms()
                self.waveform.plot(self.wave_outL)

    def make_waveforms(self):
        print('out freq: ', self.sfout)
        if self.stimpars["Stim_type"] == "click":
            self.wave_outL = self.PS.StimulusMaker(
                mode=self.stimpars["Stim_type"],
                duration=self.stimpars["click_duration"],
                freq=1000.,
                samplefreq=self.sfout,
                delay=3.,
                level=80.0,
            )
            self.wave_outR = self.wave_outL.copy()
    
    def play_waveforms(self):
        self.TrialTimer = pg.QtCore.QTimer() # get a Q timer
        self.TrialTimer.timeout.connect(self.NextTrial);
        
        self.TrialCounter = 0
        for n in self.stimpars["Reps"]:
            self.NextTrial()


# callback routine to stop timer when thread times out.
    def NextTrial(self):
        if self.debugFlag:
            print("NextTrial: entering")
        self.TrialTimer.stop()
        if self.TrialCounter <= self.stimpars["Reps"]:
            # self.statusBar().showMessage("Rep: %d of %d, ITI=%7.2f" % (self.TrialCounter+1,
            #                                                 self.totalTrials,
            #                                                 self.ITI_List[self.TrialCounter]))
            # DoneTime = self.ITI_List[self.TrialCounter] # get this before we start stimulus so stim time is included
            self.TrialTimer.start(int(self.stimpars['Per']))

            self.PS.playSound(self.wave_outL, self.wave_outR, self.sfout)

            # if self.WavePlot == True:
            #     self.plotSignal(np.linspace(0., self.Stim_Dur, len(self.wave_outL)), self.wave_outL, self.wave_outR, self.out_sampleFreq)
            # if self.AutoSave:
            #     self.AppendData(self.fn)
            self.TrialCounter = self.TrialCounter + 1
        # else:
            # self.statusBar().showMessage("Test Complete")
        print("NextTrial: exiting")
            
            # (self.ch1, self.ch2) = Sounds.retrieveInputs()

    # Initialize the updater with needed information about the plots
    #
    # updater = Updater(testMode, ecg, pltd={'plt_first': plt_first, 'plt_var': plt_var,
    #     'plt_hr': plt_hr, 'plt_current': plt_current, 'plt_RRI': plt_RRI}, ptree=ptreedata,
    #     invert=ecg.invertData)
    #
    # updater.setAllParameters(params)  # synchronize parameters with the tree

    # ptreedata.sigTreeStateChanged.connect(updater.change)  # connect parameters to their updates

    ## Start Qt event loop unless running in interactive mode.
    ## Event loop will wait for the GUI to activate the updater and start sampling.


if __name__ == "__main__":
    prog = PyABR()
