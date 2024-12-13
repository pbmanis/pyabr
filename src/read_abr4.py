import scipy
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import pyqtgraph as pg
from pyqtgraph.Qt import QtGui, QtCore
import src.laodhdf5mat as ldh5
from typing import Union
import pandas as pd
from pylibrary.tools import cprint as CP
import src.filter_util as filter_util
import src.read_abr as RA
import re

re_click = re.compile(r"[\d]{8}-[\d]{4}-[pn]{1}.txt$", re.IGNORECASE)
re_spl = re.compile(r"[\d]{8}-[\d]{4}-SPL.txt$", re.IGNORECASE)
re_khz = re.compile(r"[\d]{8}-[\d]{4}-kHz.txt$", re.IGNORECASE)
re_tone_p = re.compile(r"([\d]{8}-[\d]{4})-[p]{1}-([\d]{3,5}.[\d]{3}).txt$", re.IGNORECASE)
re_tone_n = re.compile(r"([\d]{8}-[\d]{4})-[n]{1}-([\d]{3,5}.[\d]{3}).txt$", re.IGNORECASE)

class find_datasets:
    pass


class READ_ABR4:
    def __init__(self):
        self.sample_freq = 100000.0
        self.FILT = filter_util.Utility()
        self.invert = False

        pass

    def read_dataset(
        self,
        subject: str,
        datapath: Union[Path, str],  # path to the data (.txt files are in this directory)
        datatype: str = "click",
        subdir: str="Tones", # or "Clicks"
        run: str = "20220518-1624",
        highpass: Union[float, None] = None,
        lineterm="\r",
    ):
        """
        Read a dataset, combining the positive and negative recordings,
        which are stored in separate files on disk. The waveforms are averaged
        which helps to minimize the CAP contribution.
        The waveforms are then bandpass filtered to remove the low frequency
        "rumble" and excess high-frequency noise.

        Parameters
        ----------
        run: base run name (str)
        lineterm: str
            line terminator used for this file set

        Returns
        -------
        waveform
            Waveform, as a nxm array, where n is the number of intensities,
            and m is the length of each waveform
        timebase

        """
        # handle missing files.
        if self.sample_freq is None:
            self.sample_freq = 100000.0

        if datatype == "click":
            # find click runs for this subject:
            click_runs = self.find_click_files(datapath, subject, subdir)
            for run in click_runs:
                waves, tb = self.get_clicks(datapath, subject, subdir, click_runs[run], highpass=highpass)
            return waves, tb
        elif datatype == "tone":
            self.frlist = []  # get a list of all ltone frequencies that were sampled
            tone_runs = self.find_tone_files(datapath, subject, subdir)
            allwaves = {}
            for irun, run in enumerate(tone_runs):
                waves, tb, freqs = self.get_tones(datapath, subject, subdir, tone_runs[run], highpass=highpass)
                print("get tones wave output: ", waves.shape, tb.shape, freqs)
                print("freqs: ", freqs)
                freqs = [int(float(fr)) for fr in freqs]
                for ifr, freq in enumerate(freqs):
                    if freq not in allwaves.keys():
                        print("adding freq: ", freq)
                        allwaves[int(freq)] = waves[:, ifr]  # only add if it is a new frequency
            # reconstruct waves to have all frequencies in the same array
            waves = np.zeros((len(allwaves[int(freqs[0])]), len(allwaves.keys()), len(tb)))
            self.frlist = sorted(list(allwaves.keys()))
            print("fr list: ", self.frlist)
            for ifr, freq in enumerate(self.frlist):
                waves[:, ifr] = allwaves[freq]
            print("readdataset wave shape: ", waves.shape)
            # exit()
            return waves, tb

        else:
            CP.cprint("r", f"    ABR_Reader.read_dataset: Unknown datatype: {datatype:s}")
            return None, None

    def find_click_files(self, datapath, subject, subdir):
        directory = Path(datapath, subject, subdir)
        if not directory.is_dir():
            print(f"Directory: {str(directory):s} was not found")
            exit()

        print("Directory for data found: ", str(directory))
        datafiles = list(directory.rglob(f"*.txt"))
        click_runs = {}
        for df in datafiles:
            m = re.match(re_click, df.name)
            if m is not None:
                click_runs[df.name[:13]] = {
                    "p": f"{df.name[:14]}p.txt",
                    "n": f"{df.name[:14]}n.txt",
                    "SPL": f"{df.name[:14]}SPL.txt",
                }
                # print(df.name, re.match(re_click, df.name))
                df_datetime = f"{df.name[:14]}SPL.txt"
                # print(df_datetime, re.match(re_spl, df_datetime))
        print("Found runs: ", click_runs)
        # print("datafiles: ", datafiles)
        return click_runs

    def find_tone_files(self, datapath, subject, subdir):
        directory = Path(datapath, subject, subdir)
        if not directory.is_dir():
            print("Directory: {str(directory):s} was not found")
            exit()

        print("Directory for data found: ", str(directory))
        datafiles = list(directory.rglob(f"*.txt"))
        tone_runs = {}
        runs = []
        for df in datafiles:
            mp = re.match(re_tone_p, df.name)
            mn = re.match(re_tone_n, df.name)
            print("df: ", df)
            print("mp, mn: ", mp, mn)

            if df.name[:13] not in tone_runs.keys() and (mp is not None or mn is not None):
                tone_runs[df.name[:13]] = []
            if mp is not None: 
                print('mp g: ', mp.group())
            if mn is not None:
                print('mn g: ', mn.group())
            if mp is not None:
                # print("Matched: ", df.name)
                # print(m.groups())

                tone_runs[df.name[:13]].append(
                    {
                        mp.group(2): {
                            "p": f"{df.name}",
                            # "n": f"{df.name}",
                            "SPL": f"{df.name[:14]}SPL.txt",
                            "kHz": f"{df.name[:14]}kHz.txt",
                        }
                    }
                )
            if mn is not None:
                tone_runs[df.name[:13]].append(
                    {
                        mn.group(2): {
                            "n": f"{df.name}",
                            # "n": f"{df.name}",
                            "SPL": f"{df.name[:14]}SPL.txt",
                            "kHz": f"{df.name[:14]}kHz.txt",
                        }
                    }
                )


                # print(df.name, re.match(re_click, df.name))
                df_datetime = f"{df.name[:14]}SPL.txt"
                # print(df_datetime, re.match(re_spl, df_datetime))
        # print("Found runs: ", tone_runs)
        # print("datafiles: ", datafiles)
        return tone_runs

    def get_clicks(self, datapath, subject, subdir, run, highpass:Union[float, None]=None):
        spl_file = Path(datapath, subject, subdir, run["SPL"])
        pos_file = Path(datapath, subject, subdir, run["p"])
        neg_file = Path(datapath, subject, subdir, run["n"])
        if not pos_file.is_file():
            CP.cprint("r", f"    ABR_Reader.read_dataset: Did not find pos file: {pos_file!s}")
            return None, None
        if not neg_file.is_file():
            CP.cprint("r", f"    ABR_Reader.read_dataset: Did not find neg file: {neg_file!s}")
            return None, None
        CP.cprint("c", f"    ABR_Reader.read_dataset: Reading from: {pos_file!s} and {neg_file!s}")
        exit()
        spllist = pd.read_csv(spl_file, header=None).values.squeeze()
        # print("spllist: ", spllist)
        # print(spl_file)
        # exit()
        self.dblist = spllist
        self.frlist = [0]
        posf = pd.io.parsers.read_csv(
            pos_file,
            sep=r"[\t ]+",
            lineterminator=r"[\r\n]+",  # lineterm,
            skip_blank_lines=True,
            header=None,
            names=spllist,
            engine="python",
        )
        negf = pd.io.parsers.read_csv(
            neg_file,
            sep=r"[\t ]+",
            lineterminator=r"[\r\n]+",
            skip_blank_lines=True,
            header=None,
            names=spllist,
            engine="python",
        )
        npoints = len(posf[spllist[0]])

        print(f"Number of points: {npoints:d}")
        tb = np.linspace(0, npoints * (1.0 / self.sample_freq), npoints)
        # if np.max(tb) > 25.0:
        #     u = np.where(tb < 25.0)
        #     tb = tb[u]

        npoints = tb.shape[0]
        n2 = int(npoints / 2)
        #  waves are [#db, #fr, wave]
        waves = np.zeros((len(posf.columns), len(self.frlist), n2))
        tb = tb[:n2]
        app = pg.mkQApp("summarize abr4 output")
        win = pg.GraphicsLayoutWidget(show=True, title="ABR Data Plot")
        win.resize(800, 600)
        win.setWindowTitle(f"awwww")
        symbols = ["o", "s", "t", "d", "+", "x"]
        win.setBackground("w")
        pl = win.addPlot(title=f"abr")

        for j, fr in enumerate(self.frlist):
            for i1, cn in enumerate(posf.columns):
                i = len(posf.columns) - i1 - 1
                waves[i, j] = (
                    negf[cn].values[:n2]
                    + negf[cn].values[n2:]
                    + posf[cn].values[:n2]
                    + posf[cn].values[n2:]
                ) / 4.0
                if highpass is not None:
                    print("get clicks: higpass, samplefreq: ", highpass, self.sample_freq)  
                    waves[i, j] = self.FILT.SignalFilter_HPFButter(
                        waves[i, j], HPF=highpass, samplefreq=self.sample_freq, NPole=4, bidir=True
                )

                if self.invert:
                    waves[i, j] = -waves[i, j]

        return waves, tb

    def get_tones(self, datapath, subject, subdir, tonedict, highpass:Union[float, None]=None):
        """get_tones read the tone abr data from one or more tone files taken at one time
            The data are accumulated as a 3d-array, and will be further processed by the calling

        Parameters
        ----------
        datapath : _type_
            _description_
        subject : _type_
            _description_
        tonedict : _type_
            _description_

        Returns
        -------
        _type_
            _description_
        """
        # print("tonedict: ", tonedict)
        nruns = len(tonedict)
        # print("nruns: ", nruns)
        # exit()
        waves_out = None
        freqs = []
        for irun in range(nruns):
            waves, tb, freq = self.do_one_tonerun(datapath, subject, subdir, tonedict[irun], highpass=highpass)
            if waves_out is None:
                waves_out = waves
            else:
                waves_out = np.concatenate((waves_out, waves), axis=1)
            freqs.extend(freq)
        print("wave shape: ", waves.shape, freqs)
        return waves_out, tb, freqs

    def do_one_tonerun(self, datapath, subject, subdir, runs, highpass:Union[float, None]=None):
        # combine the n and p files for the frequency in this run,
        # print("runs: ", runs)
        fr = list(runs.keys())
        print(runs)
        run = runs[fr[0]]
        print(run)
        run_p = runs[fr[0]]['p']
        run_n = runs[fr[0]]['n']
        freqs = [fr[0]]
        waves = None
        win = pg.GraphicsLayoutWidget(show=True, title="ABR Data Plot")
        win.resize(800, 600)
        win.setWindowTitle(f"awwww")
        symbols = ["o", "s", "t", "d", "+", "x"]
        win.setBackground("w")
        pl = win.addPlot(title=f"abr")
        
        # print("freqs: ", freqs)
        # Each frequency is in a different file.
        # for each file, each column is a different intensity
        # The txt files do not have any headers, so we have to know the order of the columns
        # which comes from the SPL file.
        # the khz file is redundant, because the frequency is in the filename
        for ifr, freq in enumerate(freqs):
            self.frlist.append(freq)
            print(run_p, run_n)
            spl_file = Path(datapath, subject, subdir, run["SPL"])
            khz_file = Path(datapath, subject, subdir, run["kHz"])
            pos_file = Path(datapath, subject, subdir, run_p)
            neg_file = Path(datapath, subject, subdir, run_n)
            if not pos_file.is_file():
                CP.cprint("r", f"    ABR_Reader.read_dataset: Did not find pos file: {pos_file!s}")
                return None, None, freqs
            if not neg_file.is_file():
                CP.cprint("r", f"    ABR_Reader.read_dataset: Did not find neg file: {neg_file!s}")
                return None, None, freqs
            CP.cprint("c", f"    ABR_Reader.read_dataset: Reading from:\n {pos_file!s}\n {neg_file!s}")
            exit()
            spllist = pd.read_csv(spl_file).values.squeeze()
            self.dblist = spllist
            # print("spllist: ", spllist)
            self.frlist = [0]
            posf = pd.io.parsers.read_csv(
                pos_file,
                sep=r"[\t ]+",
                lineterminator=r"[\r\n]+",  # lineterm,
                skip_blank_lines=True,
                header=None,
                names=spllist,
                engine="python",
            )
            negf = pd.io.parsers.read_csv(
                neg_file,
                sep=r"[\t ]+",
                lineterminator=r"[\r\n]+",
                skip_blank_lines=True,
                header=None,
                names=spllist,
                engine="python",
            )
            npoints = len(posf[spllist[0]])
            
            # print(f"Number of points: {npoints:d}")
            tb = np.linspace(0, npoints * (1.0 / self.sample_freq), npoints)
            npoints = tb.shape[0]
            n2 = int(npoints/2)
            #  waves are [#db, #fr, wave]
            if waves is None:
                waves = np.zeros((len(self.dblist), len(freqs), n2))
            tb0 = tb
            tb = tb[:n2]


            for i1, cn in enumerate(self.dblist):
                i = i1 # len(posf.columns) - i1 - 1
                pl.plot(tb0, posf[cn].values, pen=pg.mkPen("r", width=0.5))
                pl.plot(tb0, negf[cn].values, pen=pg.mkPen('b', width=0.5))
                waves[i, ifr] = (
                    negf[cn].values[:n2]
                    # + negf[cn].values# [n2:]
                    + posf[cn].values[:n2]
                    # + posf[cn].values# [n2:]
                ) / 2.0
                if highpass is not None:
                    print("do one tonerun: higpass, samplefreq: ", highpass, self.sample_freq)
                    waves[i, ifr] = self.FILT.SignalFilter_HPFButter(
                        waves[i, ifr], HPF=highpass, samplefreq=self.sample_freq, NPole=4, bidir=True
                    )
                # waves[i, ifr] = self.FILT.SignalFilter_LPFButter(
                #     waves[i, ifr], LPF=3000.0, samplefreq=self.sample_freq, NPole=4, bidir=True
                # )
                if self.invert:
                    waves[i, ifr] = -waves[i, ifr]
        pg.exec()
        return waves, tb0, freqs


if __name__ == "__main__":

    RAB = RA.AnalyzeABR()
    R = READ_ABR4()
    # Load the data
    # fn = "/Volumes/Pegasus_004/ManisLab_Data3/abr_data/Reggie_E/B2S_Math1cre_M_10-8_P36_WT/"
    # fn = Path(fn)

    datatype = "tone"
    if datatype == "tone":
        subdir = "Tones"
    elif datatype == "click":
        subdir = "Click"
    else:
        raise ValueError(f"Unknown datatype: {datatype:s}")

    subject = "CBA_M_p572"
    highpass = 300
    highpass = None
    maxdur = 12.0
    topdir = "/Volumes/Pegasus_002/ManisLab_Data3/abr_data/Reggie_CBA_Age/CBA_18_Months"
    fn = Path(topdir, subject, subdir)
    if not fn.is_dir():
        raise ValueError(f"File: {str(fn):s} not found")
        exit()
    w, t = R.read_dataset(
        subject=subject,
        datapath=topdir,
        subdir=subdir,
        datatype=datatype,
        highpass=highpass,
                )
    # w, t = R.read_dataset(fn, datatype="tones", subject ="")
    # R.plot_waveforms(stim_type="click", waveform=w, tb=t, fn=fn)
    print("w, t, db, fr: ", w.shape, t.shape, len(R.dblist), len(R.frlist))
    print("db: ", R.dblist)
    print(fn)
    print("R.record freq: ", R.sample_freq)
    metadata = {
        "type": "ABR4",
        "filename": str(fn),
        "subject_id": "no id",
        "age": 0,
        "sex": "ND",
        "amplifier_gain": 1e4,
        "strain": "ND",
        "weight": 0.0,
        "genotype": "ND",
        "record_frequency": 1.0 * R.sample_freq,
    }
    RAB.plot_abrs(
        abrd=w,
        tb=t,
        stim_type=datatype,
        highpass=highpass,
        scale="uV",
        dblist=R.dblist,
        frlist=R.frlist,
        metadata=metadata,
        maxdur = 12.0,
        use_matplotlib=True,
    )
