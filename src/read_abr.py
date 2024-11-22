import pickle
import matplotlib.pyplot as mpl
import platform
import pathlib
from pathlib import Path
import numpy as np
import pyqtgraph as pg
from pylibrary.plotting import plothelpers as PH
import src.read_calibration as read_calibration

use_matplotlib = False
# Check the operating system and set the appropriate path type
if platform.system() == "Windows":
    pathlib.PosixPath = pathlib.WindowsPath
else:
    pathlib.WindowsPath = pathlib.PosixPath


class AnalyzeABR:
    def __init__(self):
        self.caldata = None
        self.gain = 1e4

    def read_abr_file(self, fn):
        with open(fn, "rb") as fh:
            d = pickle.load(fh)
            self.caldata = d["calibration"]
        return d

    def show_calibration(self, fn):
        d = self.read_abr_file(fn)
        dbc = self.convert_attn_to_db(20.0, 32200)
        # print(dbc)
        read_calibration.plot_calibration(self.caldata)

    def show_calibration_history(self):
        fn = "abr_data/calibration_history"
        files = list(Path(fn).glob("frequency_MF1*.cal"))
        cal_files = []
        creation_dates = []
        for j, f in enumerate(files):
            cal_files.append(read_calibration.get_calibration_data(f))
            creation_dates.append(f.stat().st_ctime)
        app = pg.mkQApp("Calibration Data Plot")
        win = pg.GraphicsLayoutWidget(show=True, title="ABR Data Plot")
        win.resize(800, 600)
        win.setWindowTitle(f"Calibration History")
        symbols = ["o", "s", "t", "d", "+", "x"]
        win.setBackground("w")
        pl = win.addPlot(title=f"Calibration")
        # li = pg.LegendItem()
        # li.setColumnCount(2)

        # li.setParentItem(pl)
        pl.setLogMode(x=True, y=False)
        pl.addLegend()
        pl.legend.setOffset((.1, 0))
        pl.legend.setColumnCount(2)
        for cd in sorted(creation_dates):
            i = creation_dates.index(cd)
            caldata = cal_files[i]
            freqs = caldata["freqs"]
            if "2022" in caldata["date"]:
                continue
            # pl.plot(freqs, caldata['maxdb'], pen='r', name="Max SPL (0 dB Attn)")
            pl.plot(
                freqs,
                caldata["db_cs"],
                pen=pg.mkPen(i, len(cal_files), width=2),
                symbolPen=pg.mkPen(i, len(cal_files)),
                symbol=symbols[i % len(symbols)],
                symbolSize=7,
                name=f"{caldata['date']:s}",
            )
            # li.addItem(pl.legend.items[-1][1], f"{caldata['date']:s}")
            # pl.plot(freqs, caldata['db_bp'], pen='g', name=f"Measured dB SPL, attn={caldata['calattn']:.1f}, bandpass")
            # pl.plot(freqs, caldata['db_nf'], pen='b', name="Noise Floor")
            # pl.setLogMode(x=True, y=False)
        pl.setLabel("bottom", "Frequency", units="Hz")
        pl.setLabel("left", "dB SPL")
        pl.showGrid(x=True, y=True)
        # text_label = pg.LabelItem(txt, size="8pt", color=(255, 255, 255))
        # text_label.setParentItem(pl)
        # text_label.anchor(itemPos=(0.5, 0.05), parentPos=(0.5, 0.05))

        pg.exec()

    def convert_attn_to_db(self, attn, fr):
        """convert_attn_to_db converts the attenuation value at a particular
        frquency to dB SPL, based on the calibration file data

        Parameters
        ----------
        attn : float
            attenuator setting (in dB)
        fr : float
            the frequency of the stimulus (in Hz)
        """
        if self.caldata is None:
            raise ValueError(
                f"Calibration data not loaded; must load from a data file or a calibration file"
            )

        dBSPL = 0.0
        if fr in self.caldata["freqs"]:  # matches a measurement frequency
            ifr = np.where(self.caldata["freqs"] == fr)
            dBSPL = self.caldata["maxdb"][ifr]
            # print("fixed")
        else:
            # interpolate linearly between the two closest frequencies
            # first, we MUST sort the caldata and frequencies so that the freqs are in ascending
            # order, otherwise numpy gives the wrong result.
            ifr = np.argsort(self.caldata["freqs"])
            freqs = self.caldata["freqs"][ifr]
            maxdb = self.caldata["maxdb"][ifr]
            # for i, frx in enumerate(freqs):
            #     print(f"{frx:8.1f}  {maxdb[i]:8.3f}")
            dBSPL = np.interp([fr], freqs, maxdb)
        #     print("interpolated", dBSPL)
        # print(f"dBSPL = {dBSPL:8.3f} for freq={float(fr):9.2f} with {float(attn):5.1f} dB attenuation")
        dBSPL_corrected = dBSPL[0] - attn
        return dBSPL_corrected

    def average_within_traces(self, fd, i, protocol, date):
        # i is the index into the acquisition series. There is one file for each repetition of each condition.
        # the series might span a range of frequencies and intensities; these are
        # in the protocol:stimulus dictionary (dblist, freqlist)
        # we average response across traces for each intensity and frequency
        # this function specifically works when each trace has one stimulus condition (db, freq), and is
        # repeated nreps times.
        # The returned data is the average of the responses across the nreps for this (the "ith") stimulus condition
        stim_type = str(Path(fd).stem)
        print("Stim type: ", stim_type)
        if stim_type.startswith("tones"):
            name = "tonepip"
        elif stim_type.startswith("click"):
            name = "click"
        nreps = protocol["stimuli"]["nreps"]
        rec = protocol["recording"]
        fsamp = 97656.25  # samplarate, should be in the recording dict.
        nreps = protocol["stimuli"]["nreps"]
        delay = protocol["stimuli"]["delay"]
        dur = 0.010
        for n in range(nreps):  # loop over the repetitions for this specific stimulus
            fn = f"{date}_{name}_{i:03d}_{n+1:03d}.p"
            d = self.read_abr_file(Path(fd, fn))
            if n == 0:
                data = d["data"]
            else:
                data += d["data"]
            if n == 0:
                tb = np.arange(0, len(data) / fsamp, 1 / fsamp)
        data = data / nreps
        # tile the traces.
        # first interpolate to 100 kHz
        # If you don't do this, the blocks will precess in time against
        # the stimulus, which is timed on a 500 kHz clock.
        # It is an issue because the TDT uses an odd frequency clock...

        trdur = len(data) / fsamp
        newrate = 1e5
        tb100 = np.arange(0, trdur, 1.0 / newrate)

        one_response = int(0.025 * newrate)
        arraylen = one_response * protocol["stimuli"]["nstim"]

        abr = np.interp(tb100, tb, data)
        sub_array = np.split(abr[:arraylen], protocol["stimuli"]["nstim"])
        sub_array = np.mean(sub_array, axis=0)
        tb = tb[:one_response]
        return sub_array, tb

    def average_across_traces(self, fd, i, protocol, date):
        """average_across_traces for abrs with multiple stimuli in a trace.
        This function averages the responses across multiple traces.
        and returns a list broken down by the individual traces.

        Parameters
        ----------
        fd : _type_
            _description_
        i : _type_
            _description_
        protocol : _type_
            _description_
        date : _type_
            _description_
        stim_type : _type_
            _description_

        Returns
        -------
        _type_
            _description_
        """
        nreps = protocol["stimuli"]["nreps"]
        rec = protocol["recording"]
        fsamp = 97656.25  # sample rate, should be in the recording dict.
        delay = protocol["stimuli"]["delay"]
        dur = 0.010
        dataset = []
        dblist = protocol["stimuli"]["dblist"]
        frlist = protocol["stimuli"]["freqlist"]

        stim_type = str(Path(fd).stem)
        for n in range(nreps):
            fn = f"{date}_{stim_type}_{i:03d}_{n+1:03d}.p"
            d = self.read_abr_file(Path(fd, fn))

            if n == 0:
                data = d["data"]
            else:
                data += d["data"]
            if n == 0:
                print("sample rate: ", fsamp)
                print("number of data points: ", len(data))
                tb = np.arange(0, len(data) / fsamp, 1 / fsamp)

        data = data / nreps
        # tile the traces.
        # first linspace to 100 kHz
        trdur = len(data) / fsamp
        newrate = 1e5
        tb100 = np.arange(0, trdur, 1.0 / newrate)
        abr = np.interp(tb100, tb, data)
        it0 = int(protocol["stimuli"]["delay"] * newrate)

        abr = abr[it0:]
        one_response = int(protocol["stimuli"]["stimulus_period"] * newrate)
        print(protocol["stimuli"])
        if isinstance(protocol["stimuli"]["freqlist"], str):
            frlist = eval(protocol["stimuli"]["freqlist"])
        if isinstance(protocol["stimuli"]["dblist"], str):
            dblist = eval(protocol["stimuli"]["dblist"])
        arraylen = one_response * protocol["stimuli"]["nstim"]
        if stim_type in ["click", "tonepip"]:
            nsplit = protocol["stimuli"]["nstim"]
        elif stim_type in ["interleaved_plateau"]:
            nsplit = int(len(frlist) * len(dblist))
        arraylen = one_response * nsplit
        # print(len(frlist), len(dblist))
        # print("one response: ", one_response)
        # print("arraylen: ", arraylen)
        # print("nsplit: ", nsplit)
        # print("arranlen/nsplit: ", float(arraylen)/nsplit)
        # print("len data: ", len(data), len(abr), nsplit*one_response)
        sub_array = np.split(abr[:arraylen], nsplit)
        # print(len(sub_array))
        abr = np.array(sub_array)
        tb = tb[:one_response]
        print("abr shape: ", abr.shape)
        stim = np.meshgrid(frlist, dblist)
        print(stim)
        return abr, tb, stim

    def show_stimuli(self, fn):
        d = self.read_abr_file(fn)
        stims = list(d["stimuli"].keys())
        wave = d["stimuli"][stims[0]]["sound"]
        pg.plot(wave)
        pg.exec()

    def read_and_average_abr_files(self, fn):
        d = self.read_abr_file(fn)
        # print(d["protocol"])
        stim_type = d["protocol"]["protocol"]["stimulustype"]
        fd = Path(fn).parent
        fns = Path(fd).glob("*.p")
        # break the filenames into parts.
        # The first part is the date, the second part is the stimulus type,
        # the third part is the index into the stimulus array
        # the fourth part is the repetition number for a given stimulus
        protocol = d["protocol"]
        rec = protocol["recording"]
        dblist = protocol["stimuli"]["dblist"]
        frlist = protocol["stimuli"]["freqlist"]
        if isinstance(dblist, str):
            dblist = eval(dblist)
        if isinstance(frlist, str):
            frlist = eval(frlist)
        ndb = len(dblist)
        nreps = protocol["stimuli"]["nreps"]
        delay = protocol["stimuli"]["delay"]
        dur = 0.010

        file_parts = Path(fn).stem.split("_")
        # print(file_parts)
        date = file_parts[0]
        stim_type = file_parts[1]
    
        if len(frlist) == 0:
            frlist = [1]
        if stim_type in ["click", "tonepip"]:
            n = 0
            for i, db in enumerate(dblist):
                for j, fr in enumerate(frlist):
                    x, tb = average_within_traces(
                        fd,
                        n,
                        protocol,
                        date,
                    )
                    if i == 0 and j == 0:
                        abrd = np.zeros((len(dblist), len(frlist), len(x)))
                    abrd[i, j] = x
                    n += 1

            if stim_type in ["interleaved"]:
                n = 0
                abrd, tb, stim = self.average_across_traces(fd, n, protocol, date)
                # print(len(frlist), len(dblist))
                abrd = abrd.reshape(len(dblist), len(frlist), -1)
            if use_matplotlib:
                P = PH.regular_grid(
                    cols=len(frlist),
                    rows=len(dblist),
                    order="rowsfirst",
                    figsize=(2 * len(frlist), 1.0 * len(dblist)),
                    verticalspacing=0.0,
                    horizontalspacing=0.02,
                )
                ax = P.axarr
                n = 0
                for i, db in enumerate(dblist):
                    for j, fr in enumerate(frlist):  # enumerate(abrd.keys()):
                        ax = P.axarr[len(dblist) - i - 1, j]
                        # ax.plot(tb, abrd[(dblist[i], frlist[j])])
                        if stim_type in ["click", "tonepip"]:

                            ax.plot(tb * 1e3, abrd[i, j], clip_on=False)
                        else:
                            ax.plot(tb * 1e3, abrd[len(dblist) - i - 1, j], clip_on=False)
                        if i == len(dblist) - 1:
                            ax.set_title(f"{db} dB, {fr} Hz")
                    if i == 0:
                        ax.set_xlabel("Time (s)")
                    if j == 0:
                        ax.set_ylabel("Amplitude")
                    ax.set_ylim(-50, 50)
                    PH.noaxes(ax)
                    if i == 0 and j == 0:
                        PH.calbar(ax, calbar=[0, -20, 2, 10], scale=[1.0, 1.0], xyoffset=[0.05, 0.1],)
                    PH.referenceline(ax, linewidth=0.5)
                    n += 1
                    
                # mpl.tight_layout()
                mpl.show()
        else: # use pyqtgraph
            app = pg.mkQApp("ABR Data Plot")
            win = pg.GraphicsLayoutWidget(show=True, title="ABR Data Plot")
            win.resize(1000,1000)
            win.setWindowTitle(f"File: {str(Path(fn).parent)}")
            plid = []
            if len(frlist) == 0:
                frlist = [1]
            col = 0
            print("stim_type: ", stim_type)
            if stim_type not in ["clicks", "click"]:
                for i, db in enumerate(dblist):
                    row = i # int(i/5)
                    for j, fr in enumerate(frlist):
                        col=j
                        pl = win.addPlot(title=f"{db} dB, {fr} Hz", col=col, row= row) # i % 5)
                        if stim_type in ["click", "tonepip"]:
                            pl.plot(tb*1e3, abrd[i, j]/amplifier_gain, pen=pg.mkPen(j, len(dblist), width=2), clipToView=True)
                        else:
                            pl.plot(tb*1e3, abrd[len(dblist)-i-1, j]/amplifier_gain, pen=pg.mkPen(j, len(dblist), width=2),  clipToView=True)
                        # pl.showGrid(x=True, y=True)
                        if j == 0:
                            pl.setLabel('left', "Amplitude", units='mV')
                        if i == 0:
                            pl.setLabel('bottom', "Time", units='s')
                        pl.setYRange(-2e-6, 2e-6)


            else:
                v0 = 0
                v=[]
                for i, db in enumerate(dblist):
                    if i == 0:
                        pl = win.addPlot(title=f"{db} dB, {fr} Hz") # i % 5)
                    pl.plot(tb*1e3, -v0 + abrd[i, j]/amplifier_gain, pen=pg.mkPen(pg.intColor(i, hues=len(dblist)), width=2), clipToView=True)
                    v0 += 1e-6*amplifier_gain
                    v.append(v0)
                    # pl.showGrid(x=True, y=True)
                    pl.setLabel('left', "Amplitude", units='mV')
                    pl.setLabel('bottom', "Time", units='s')
                    label = pg.LabelItem(f"{db:.1f}", size="11pt", color="#99aadd")
                    label.setParentItem(pl)
                    label.anchor(itemPos=(0.05, -v0*180), parentPos=(0.1 , 0))
                    # pl.setYRange(-2e-6, 2e-6)
                    plid.append(pl)
                for i, db in enumerate(dblist):
                    label = pg.LabelItem(f"{db:.1f}", size="11pt", color="#99aadd")
                    label.setParentItem(pl)
                    label.anchor(itemPos=(0.05, -v[i]*200), parentPos=(0.1 , 0))
                    # win.nextRow()
                    for j, fr in enumerate(frlist):
                        ax.set_title(f"{self.convert_attn_to_db(db, fr)} dBSPL, {fr} Hz")
                        if i == 0:
                            ax.set_xlabel("Time (s)")
                        if j == 0:
                            ax.set_ylabel("Amplitude")
                        ax.set_ylim(-50, 50)
                        PH.noaxes(ax)
                        if i == 0 and j == 0:
                            PH.calbar(
                                ax,
                                calbar=[0, -20, 2, 10],
                                scale=[1.0, 1.0],
                                xyoffset=[0.05, 0.1],
                            )
                        PH.referenceline(ax, linewidth=0.5)
                        n += 1


            # else:  # use pyqtgraph
            #     app = pg.mkQApp("ABR Data Plot")
            #     win = pg.GraphicsLayoutWidget(show=True, title="ABR Data Plot")
            #     win.resize(1000, 1000)
            #     win.setWindowTitle(f"File: {str(Path(fn).parent)}")
            #     plid = []
            #     if len(frlist) == 0:
            #         frlist = [1]
            #     for i, db in enumerate(dblist):
            #         for j, fr in enumerate(frlist):
            #             pl = win.addPlot(
            #                 title=f"{self.convert_attn_to_db(db, fr):5.1f} dB, {fr:8.1f} Hz"
            #             )
            #             if stim_type in ["click", "tonepip"]:
            #                 pl.plot(
            #                     tb * 1e3,
            #                     abrd[i, j] / self.gain,
            #                     pen=pg.mkPen(j, len(dblist), width=2),
            #                     clipToView=True,
            #                 )
            #             else:
            #                 pl.plot(
            #                     tb * 1e3,
            #                     abrd[len(dblist) - i - 1, j] / self.gain,
            #                     pen=pg.mkPen(j, len(dblist), width=2),
            #                     clipToView=True,
            #                 )
            #             # pl.showGrid(x=True, y=True)
            #             if j == 0:
            #                 pl.setLabel("left", "Amplitude", units="mV")
            #             if i == 0:
            #                 pl.setLabel("bottom", "Time", units="s")
            #             pl.setYRange(-5e-6, 5e-6)
            #             plid.append(pl)
            #         win.nextRow()

            pg.exec()

    # Enable antialiasing for prettier plots
    pg.setConfigOptions(antialias=True)


if __name__ == "__main__":
    fn = "abr_data/2024-11-13/clicks"
    fn = "abr_data/2024-11-15-B/interleaved_plateau"
    files = list(Path(fn).glob("*.p"))
    print(str(files[0]))
    print(files[0].is_file())

    AR = AnalyzeABR()
    # AR.show_calibration(files[0])
    AR.show_calibration_history()
    exit()
    d = AR.read_abr_file(str(files[0]))
    print(d["stimuli"].keys())

    # AR.show_stimuli(files[0])
    AR.read_and_average_abr_files(str(files[0]))
