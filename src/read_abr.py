import pickle
import datetime
import matplotlib.pyplot as mpl
import platform
import pathlib
from pathlib import Path
from pylibrary.tools import cprint as CP
from typing import Union
import numpy as np
import pyqtgraph as pg
import plothelpers as mpl_PH
import src.read_calibration as read_calibration
import src.filter_util as filter_util
import src.parse_ages as PA
import src.read_abr4 as read_abr4

use_matplotlib = True
from pylibrary.plotting import plothelpers as PH
from matplotlib.backends.backend_pdf import PdfPages

# Check the operating system and set the appropriate path type
if platform.system() == "Windows":
    pathlib.PosixPath = pathlib.WindowsPath
else:
    pathlib.WindowsPath = pathlib.PosixPath


class AnalyzeABR:
    def __init__(self):
        self.caldata = None
        self.gain = 1e4
        self.fsamp = 24414.0625
        self.FILT = filter_util.Utility()
        self.frequencies = []
        self.hide_treatment = False

        # 24414.0625
    def set_hide_treatment(self, hide_treatment: bool):
        self.hide_treatment = hide_treatment

    def read_abr_file(self, fn):
        with open(fn, "rb") as fh:
            d = pickle.load(fh)
            self.caldata = d["calibration"]
            self.fsamp = d["record_frequency"]
            self.fsamp = self.check_fsamp(d)
            # Trim the data array to remove the delay to the stimulus.
            # to be consistent with the old ABR4 program, we leave
            # the first 1 msec prior to the stimulus in the data array.
            delay = float(d["protocol"]["stimuli"]["delay"])
            i_delay = int((delay - 0.001) * self.fsamp)
            # print("delay: ", delay, i_delay)
            d["data"] = d["data"][i_delay:]
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
        pl.legend.setOffset((0.1, 0))
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

    def average_within_traces(
        self, fd, i, protocol, date, high_pass_filter: Union[float, None] = None
    ):
        # i is the index into the acquisition series. There is one file for each repetition of each condition.
        # the series might span a range of frequencies and intensities; these are
        # in the protocol:stimulus dictionary (dblist, freqlist)
        # we average response across traces for each intensity and frequency
        # this function specifically works when each trace has one stimulus condition (db, freq), and is
        # repeated nreps times.
        # The returned data is the average of the responses across the nreps for this (the "ith") stimulus condition
        stim_type = str(Path(fd).stem)
        print("Stim type: ", stim_type)
        print("Protocol: ", protocol)
        if stim_type.lower().startswith("tone"):
            name = "tonepip"
        elif stim_type.lower().startswith("interleaved"):
            name = "interleaved_plateau"
        elif stim_type.lower().startswith("click"):
            name = "click"
        nreps = protocol["stimuli"]["nreps"]
        rec = protocol["recording"]

        nreps = protocol["stimuli"]["nreps"]
        missing_reps = []
        for n in range(nreps):  # loop over the repetitions for this specific stimulus
            fn = f"{date}_{name}_{i:03d}_{n+1:03d}.p"
            if not Path(fd, fn).is_file():
                missing_reps.append(n)
                continue
            d = self.read_abr_file(Path(fd, fn))
            if n == 0:
                data = d["data"]
            else:
                data += d["data"]
            if n == 0:
                tb = np.linspace(0, len(data) / self.fsamp, len(data))
        if len(missing_reps) > 0:
            CP.cprint("r", f"Missing {len(missing_reps)} reps for {name} {i}", "red")
        data = data / (nreps - len(missing_reps))
        if high_pass_filter is not None:
            print("average within traces hpf: ", high_pass_filter, self.fsamp)
            data = self.FILT.SignalFilter_HPFButter(
                data, high_pass_filter, self.fsamp, NPole=4, bidir=True
            )
        # tile the traces.
        # first interpolate to 100 kHz
        # If you don't do this, the blocks will precess in time against
        # the stimulus, which is timed on a 500 kHz clock.
        # It is an issue because the TDT uses an odd frequency clock...

        trdur = len(data) / self.fsamp
        newrate = 1e5
        tb100 = np.arange(0, trdur, 1.0 / newrate)

        one_response = int(0.025 * newrate)
        arraylen = one_response * protocol["stimuli"]["nstim"]

        abr = np.interp(tb100, tb, data)
        sub_array = np.split(abr[:arraylen], protocol["stimuli"]["nstim"])
        sub_array = np.mean(sub_array, axis=0)
        tb = tb[:one_response]
        return sub_array, tb

    def average_across_traces(
        self, fd, i, protocol, date, high_pass_filter: Union[float, None] = None
    ):
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

        dblist = protocol["stimuli"]["dblist"]
        frlist = protocol["stimuli"]["freqlist"]
        stim_type = str(Path(fd).stem)
        # print("Stim type: ", stim_type)
        if stim_type.lower().startswith("interleaved"):
            stim_type = "interleaved_plateau"
        missing_reps = []
        ndata = 0
        for n in range(nreps):
            fn = f"{date}_{stim_type}_{i:03d}_{n+1:03d}.p"
            if not Path(fd, fn).is_file():
                missing_reps.append(n)
                continue
            d = self.read_abr_file(Path(fd, fn))

            if ndata == 0:
                data = d["data"]
            else:
                data += d["data"]
            if ndata == 0:
                # print("original sample rate: ", self.fsamp)
                # print("number of data points: ", len(data))
                tb = np.arange(0, len(data) / self.fsamp, 1.0 / self.fsamp)
            ndata += 1
        if len(missing_reps) > 0:
            CP.cprint("r", f"Missing {len(missing_reps)} reps for {fn!s}", "red")
        data = data / (nreps - len(missing_reps))
        #  high pass filter at 300 Hz
        # data = self.FILT.SignalFilter_LPFButter(data, LPF=3000, samplefreq=self.fsamp, NPole=4)
        if high_pass_filter is not None:
            print("average across traces hpf: ", high_pass_filter, self.fsamp)
            data = self.FILT.SignalFilter_HPFButter(
                data, HPF=high_pass_filter, samplefreq=self.fsamp, NPole=4, bidir=True
            )
        # tile the traces.
        # first linspace to 100 kHz
        trdur = len(data) / self.fsamp
        newrate = 1e5
        tb = np.linspace(0, trdur, len(data))
        # f, ax = mpl.subplots(2,1)
        # ax[0].plot(tb, data)

        tb100 = np.arange(0, trdur, 1.0 / newrate)
        # print('len tb100: ', len(tb100), np.mean(np.diff(tb100)))
        abr = np.interp(tb100, tb, data)
        # ax[0].plot(tb100, abr, '--r', linewidth=0.5)
        # mpl.show()
        # exit()

        one_response_100 = int(protocol["stimuli"]["stimulus_period"] * newrate)
        print(protocol["stimuli"])
        if isinstance(protocol["stimuli"]["freqlist"], str):
            frlist = eval(protocol["stimuli"]["freqlist"])
        if isinstance(protocol["stimuli"]["dblist"], str):
            dblist = eval(protocol["stimuli"]["dblist"])
        arraylen = one_response_100 * protocol["stimuli"]["nstim"]
        if stim_type.lower() in ["click", "tonepip"]:
            nsplit = protocol["stimuli"]["nstim"]
        elif stim_type.lower() in ["interleaved_plateau"]:
            nsplit = int(len(frlist) * len(dblist))
        else:
            raise ValueError(f"Stimulus type {stim_type} not recognized")
        arraylen = one_response_100 * nsplit
        # print(len(frlist), len(dblist))
        # print("one response: ", one_response_100)
        # print("nsplit: ", nsplit)
        # print("arranlen/nsplit: ", float(arraylen) / nsplit)
        # print("len data: ", len(data), len(abr), nsplit * one_response_100)
        sub_array = np.split(abr[:arraylen], nsplit)
        # print(len(sub_array))
        abr = np.array(sub_array)
        tb = tb100[:one_response_100]

        # print("abr shape: ", abr.shape, "max time: ", np.max(tb))
        stim = np.meshgrid(frlist, dblist)
        self.fsamp = newrate
        # print(stim)
        return abr, tb, stim

    def show_stimuli(self, fn):
        d = self.read_abr_file(fn)
        stims = list(d["stimuli"].keys())
        wave = d["stimuli"][stims[0]]["sound"]
        pg.plot(wave)
        pg.exec()

    def plot_abrs(
        self,
        abr_data: np.ndarray,
        tb: np.ndarray,
        scale: str = "V",
        acquisition: str = "pyabr3",
        V_stretch: float = 1.0,
        highpass: Union[float, None] = None,
        stim_type: str = "click",
        dblist: Union[list, None] = None,
        frlist: Union[list, None] = None,
        maxdur: float = 14.0,
        metadata: dict = {},
        use_matplotlib: bool = True,
        live_plot: bool = False,
        pdf: Union[object, None] = None,
    ):

        amplifier_gain = metadata["amplifier_gain"]
        if scale == "uV":
            # amplifier gain has already been accountd for.
            added_gain = 1.0
        elif scale == "V":
            added_gain = 1e6  # convert to microvolts
        else:
            raise ValueError(f"Scale {scale} not recognized, must be 'V' or 'uV'")

        if len(frlist) == 0 or stim_type == "click":
            frlist = [0]
            ncols = 1
            width = 5  # (1.0 / 3.0) * len(dblist) * ncols
            height = 1.0 * len(dblist)
            lmar = 0.15
        else:
            ncols = len(frlist)
            width = 2.0 * ncols
            height = 1.0 * len(dblist)
            lmar = 0.1

        if height > 10.0:
            height = 10.0 * (10.0 / height)
        if use_matplotlib:
            P = mpl_PH.regular_grid(
                cols=ncols,
                rows=len(dblist),
                order="rowsfirst",
                figsize=(width, height),
                verticalspacing=0.01,
                horizontalspacing=0.03,
                margins={
                    "leftmargin": lmar,
                    "rightmargin": 0.05,
                    "topmargin": 0.15,
                    "bottommargin": 0.10,
                },
            )
            fn = metadata["filename"]
            subname = ""
            if metadata["type"] == "ABR4":
                filename = str(Path(fn).name)
                if filename in ["Click", "Clicks", "Tone", "Tones"]:
                    filename = str(Path(fn).parent.name)
                    parentname = str(Path(fn).parent.parent.name)
                    subname = "/".join(Path(fn).parts[-3:])
                else:
                    filename = str(Path(fn).parent)
                    parentname = str(Path(fn).parent.parent)
                    subname = "/".join(Path(fn).parts[-2:])
            else:
                filename = str(Path(fn).parent)
                parentname = str(Path(fn).parent.parent)

            subject_id = metadata["subject_id"]
            age = PA.ISO8601_age(metadata["age"])
            sex = metadata["sex"]
            amplifier_gain = metadata["amplifier_gain"]
            strain = metadata["strain"]
            weight = metadata["weight"]
            genotype = metadata["genotype"]
            rec_freq = metadata["record_frequency"]
            if highpass is None:
                hpftext = "Off"
            else:
                hpftext = f"{highpass:5.1f} Hz"
            title_file_name = filename

            if self.hide_treatment:
                # hide the last 
                if acquisition == "ABR4":
                    tf_parts = list(Path(fn).parts)
                    fnp = tf_parts[-2]
                    fnsplit = fnp.split("_")
                    fnsplit[-1] = "*"
                    tf_parts[-2] = '_'.join([x for x in fnsplit])
                    title_file_name = tf_parts[-2]
                    page_file_name = str(Path(*tf_parts))
                elif acquisition == "pyabr3":
                    tf_parts = list(Path(fn).parts)
                    fnp = tf_parts[-3]
                    fnsplit = fnp.split("_")
                    fnsplit[-1] = "*"
                    tf_parts[-3] = '_'.join([x for x in fnsplit])
                    page_file_name = str(Path(*tf_parts[:-1]))
                    title_file_name = tf_parts[-3]

            title = f"\n{title_file_name!s}\n"
            title += f"Stimulus: {stim_type}, Amplifier Gain: {amplifier_gain}, Fs: {rec_freq}, HPF: {hpftext:s}, Acq: {acquisition:s}\n"
            title += f"Subject: {subject_id:s}, Age: {age:s} Sex: {sex:s}, Strain: {strain:s}, Weight: {weight:.2f}, Genotype: {genotype:s}"
            # if acquisition == "pyabr3":
            #     print("title: ", title)
            #     print(self.hide_treatment)
            #     exit()

            P.figure_handle.suptitle(title, fontsize=8)
            ax = P.axarr
            v_min = 0.0
            v_max = 0.0
            n = 0
            colors = [
                "xkcd:raspberry",
                "xkcd:red",
                "xkcd:orange",
                "xkcd:golden yellow",
                "xkcd:green",
                "xkcd:blue",
                "xkcd:purple",
                "xkcd:bright violet",
            ]
            click_colors = [
                "xkcd:azure",
                "xkcd:lightblue",
                "xkcd:purple",
                "xkcd:orange",
                "xkcd:red",
                "xkcd:green",
                "xkcd:golden yellow",
            ]
            n_click_colors = len(click_colors)
            refline_ax = []
            stack_dir = "up"
            if metadata["type"] == "ABR4":
                stack_dir = "up"
            elif metadata["type"] == "pyabr3" and stim_type.lower().startswith("click"):
                stack_dir = "down"
            elif metadata["type"] == "pyabr3" and (
                stim_type.lower().startswith("tone") or stim_type.lower().startswith("interleaved")
            ):
                stack_dir = "up"
            else:
                raise ValueError(
                    f"Stimulus type {stim_type} not recognized togethger with data type {metadata['type']}"
                )
            for j, fr in enumerate(range(ncols)):  # enumerate(abr_data.keys()):
                for i, db in enumerate(dblist):

                    if stack_dir == "up":
                        ax = P.axarr[len(dblist) - i - 1, j]
                    else:
                        ax = P.axarr[i, j]

                    npts = len(abr_data[i, j])
                    n_disp_pts = int(maxdur * 1e-3 * rec_freq)  # maxdur is in msec.
                    if n_disp_pts < npts:
                        npts = n_disp_pts
                    # print("added_gain: ", added_gain)
                    plot_data = added_gain * abr_data[i, j] / amplifier_gain
                    if stim_type in ["click", "tonepip"]:
                        ax.plot(
                            (tb[:npts]) * 1e3,
                            plot_data[:npts],
                            color=click_colors[i % n_click_colors],
                            linewidth=1,
                            clip_on=False,
                        )
                    else:
                        ax.plot(
                            (tb[:npts]) * 1e3,
                            plot_data[:npts],
                            color=colors[j],
                            clip_on=False,
                        )
                    if ax not in refline_ax:
                        PH.referenceline(ax, linewidth=0.5)
                        refline_ax.append(ax)
                    # print(dir(ax))
                    ax.set_facecolor(
                        "#ffffff00"
                    )  # background will be transparent, allowing traces to extend into other axes
                    ax.set_xlim(0, maxdur)
                    # let there be an axis on one trace (at the bottom)

                    if stack_dir == "up":
                        if i == len(dblist) - 1:
                            if ncols > 1:
                                ax.set_title(f"{frlist[j]} Hz")
                            else:
                                ax.set_title("Click")
                            PH.noaxes(ax)
                        elif i == 0:
                            PH.nice_plot(ax, direction="outward", ticklength=3)
                            ax.set_xlabel("Time (ms)")
                            ticks = np.arange(0, maxdur, 2)
                            ax.set_xticks(ticks, [f"{int(k):d}" for k in ticks])
                        else:
                            PH.noaxes(ax)

                    elif stack_dir == "down":
                        if i == 0:
                            if ncols > 1:
                                ax.set_title(f"{frlist[j]} Hz")
                            else:
                                ax.set_title("Click")
                            PH.noaxes(ax)
                        elif i == len(dblist) - 1:
                            PH.nice_plot(ax, direction="outward", ticklength=3)
                            ticks = np.arange(0, maxdur, 2)
                            ax.set_xticks(ticks, [f"{int(k):d}" for k in ticks])
                            ax.set_xlabel("Time (ms)")
                        else:
                            PH.noaxes(ax)

                    if j == 0:
                        ax.set_ylabel(
                            f"{int(float(dblist[i])):d}dBSPL ",
                            fontsize=8,
                            labelpad=0,
                            rotation=0,
                            ha="right",
                            va="center",
                        )
                    ax.set_xlim(0, np.max(tb[:npts]) * 1e3)

                # ax.set_xticks([1, 3, 5, 7, 9], minor=True)

                n += 1
                if np.max(plot_data[:npts]) > v_max:
                    v_max = np.max(plot_data[:npts])
                if np.min(plot_data[:npts]) < v_min:
                    v_min = np.min(plot_data[:npts])

            if metadata["type"] == "pyabr3" and stim_type.lower().startswith("click"):
                V_stretch = 10.0 * V_stretch

            amax = np.max([-v_min, v_max]) * V_stretch
            if amax < 0.5:
                amax = 0.5
            # print(P.axarr.shape, len(dblist), len(frlist))

            for i in range(len(dblist)):
                for j in range(len(frlist)):
                    P.axarr[i, j].set_ylim(-amax, amax)
                    # PH.referenceline(ax, linewidth=0.5)
            mpl.text(
                0.96,
                0.01,
                s=datetime.datetime.now(),
                fontsize=6,
                ha="right",
                transform=P.figure_handle.transFigure,
            )
            mpl.text(
                0.02,
                0.01,
                s=f"{page_file_name:s}",
                fontsize=5,
                ha="left",
                transform=P.figure_handle.transFigure,
            )

            if live_plot:
                mpl.show()
            else:
                # print(P.figure_handle)
                pdf.savefig(P.figure_handle)
                mpl.close()
            # mpl.tight_layout()
            # mpl.show()

        else:  # use pyqtgraph
            app = pg.mkQApp("ABR Data Plot")
            win = pg.GraphicsLayoutWidget(show=True, title="ABR Data Plot")
            win.resize(1200, 1000)
            win.setWindowTitle(f"File: {str(Path(fn).parent)}")
            win.setBackground("w")

            lw = pg.LayoutWidget(parent=win)
            lw.addLabel(text="Hi there", row=0, col=0, rowspan=1, colspan=len(frlist))
            lw.nextRow()

            plid = []
            if len(frlist) == 0:
                frlist = [1]
            col = 0

            print("stim_type (in pg plotting)", stim_type)
            if stim_type not in ["clicks", "click"]:  # this is for tones/interleaved, etc.
                ref_set = False
                v_min = 0
                v_max = 0
                for i, db in enumerate(dblist):
                    row = i  # int(i/5)
                    for j, fr in enumerate(frlist):
                        col = j
                        # if tb is None:
                        #     npts = len(abr_data[i, j])
                        #     tb = np.linspace(0, npts / rec_freq, npts)

                        pl = win.addPlot(
                            title=f"{dblist[-i-1]} dB, {fr} Hz", col=col, row=row
                        )  # i % 5)
                        if not ref_set:
                            ref_ax = pl
                            ref_set = True
                        plot_data = 1e6 * abr_data[len(dblist) - i - 1, j] / amplifier_gain
                        lpd = len(plot_data)
                        if stim_type in ["click", "tonepip"]:
                            pl.plot(
                                tb[:lpd] * 1e3,
                                plot_data,
                                pen=pg.mkPen(j, len(dblist), width=2),
                                clipToView=True,
                            )
                        else:
                            pl.plot(
                                tb[:lpd] * 1e3,
                                plot_data,
                                pen=pg.mkPen(j, len(dblist), width=2),
                                clipToView=True,
                            )
                        pl.plot(
                            tb[:lpd] * 1e3,
                            np.zeros_like(plot_data),
                            pen=pg.mkPen(
                                "grey", linetype=pg.QtCore.Qt.PenStyle.DashLine, width=0.33
                            ),
                            clipToView=True,
                        )
                        # pl.showGrid(x=True, y=True)
                        if j == 0:
                            pl.setLabel("left", "Amp", units="uV")
                        if i == len(dblist) - 1:
                            pl.setLabel("bottom", "Time", units="ms")
                        pl.setYRange(-3.0, 3.0)
                        pl.setXRange(0, 10)
                        if ref_set:
                            pl.setXLink(ref_ax)
                            pl.setYLink(ref_ax)
                        if np.max(plot_data) > v_max:
                            v_max = np.max(plot_data)
                        if np.min(plot_data) < v_min:
                            v_min = np.min(plot_data)

                ref_ax.setYRange(v_min * V_stretch, v_max * V_stretch)

            else:  # pyqtgraph
                v0 = 0
                v = []
                for i, db in enumerate(dblist):
                    if i == 0:
                        pl = win.addPlot(title=f"{db} dB, {fr} Hz")  # i % 5)
                    pl.plot(
                        tb * 1e3,
                        -v0 + abr_data[i, j] / amplifier_gain,
                        pen=pg.mkPen(pg.intColor(i, hues=len(dblist)), width=2),
                        clipToView=True,
                    )
                    v0 += 1e-6 * amplifier_gain
                    v.append(v0)
                    # pl.showGrid(x=True, y=True)
                    pl.setLabel("left", "Amplitude", units="mV")
                    pl.setLabel("bottom", "Time", units="s")
                    label = pg.LabelItem(f"{db:.1f}", size="11pt", color="#99aadd")
                    label.setParentItem(pl)
                    label.anchor(itemPos=(0.05, -v0 * 180), parentPos=(0.1, 0))
                    # pl.setYRange(-2e-6, 2e-6)
                    plid.append(pl)
                for i, db in enumerate(dblist):
                    label = pg.LabelItem(f"{db:.1f}", size="11pt", color="#99aadd")
                    label.setParentItem(pl)
                    label.anchor(itemPos=(0.05, -v[i] * 200), parentPos=(0.1, 0))
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

            pg.exec()

        # Enable antialiasing for prettier plots
        pg.setConfigOptions(antialias=True)

    def check_fsamp(self, d):
        if d["record_frequency"] is None:
            d["record_frequency"] = 24414.0625  # 97656.25
        return d["record_frequency"]

    def read_and_average_abr_files(
        self,
        filename,
        amplifier_gain=1e4,
        scale: str = "V",
        high_pass_filter: Union[float, None] = None,
        maxdur: Union[float, None] = None,
        pdf: Union[object, None] = None,
    ):
        d = self.read_abr_file(filename)
        print("     Read and average abrs")
        # print(d["protocol"])
        # print("d keys: ", d.keys())
        self.fsamp = self.check_fsamp(d)
        if maxdur is None:
            maxdur = 25.0
        stim_type = d["protocol"]["protocol"]["stimulustype"]
        fd = Path(filename).parent
        fns = Path(fd).glob("*.p")
        # break the filenames into parts.
        # The first part is the date,
        # the second part is the stimulus type,
        # the third part is the index into the stimulus array
        # the fourth part is the repetition number for a given stimulus
        protocol = d["protocol"]
        rec = protocol["recording"]
        # print("rec: ", rec)
        dblist = protocol["stimuli"]["dblist"]
        frlist = protocol["stimuli"]["freqlist"]
        if isinstance(dblist, str):
            dblist = eval(dblist)
        if isinstance(frlist, str):
            frlist = eval(frlist)

        subject_id = d["subject_data"]["Subject ID"]
        if d["subject_data"]["Age"] != "":
            age = PA.ISO8601_age(d["subject_data"]["Age"])
        else:
            age = 0
        sex = d["subject_data"]["Sex"]
        strain = d["subject_data"]["Strain"]
        if d["subject_data"]["Weight"] != "":
            weight = float(d["subject_data"]["Weight"])
        else:
            weight = 0.0
        genotype = d["subject_data"]["Genotype"]

        file_parts = Path(filename).stem.split("_")
        # print(file_parts)
        date = file_parts[0]
        stim_type = file_parts[1]
        # print("stim type(before...): ", stim_type)
        if len(frlist) == 0:
            frlist = [1]
        if stim_type in ["click", "tonepip"]:
            n = 0
            for i, db in enumerate(dblist):
                for j, fr in enumerate(frlist):
                    x, tb = self.average_within_traces(
                        fd,
                        n,
                        protocol,
                        date,
                        high_pass_filter=high_pass_filter,
                    )
                    if i == 0 and j == 0:
                        abr_data = np.zeros((len(dblist), len(frlist), len(x)))
                    abr_data[i, j] = x
                    n += 1

        # print("stim type:: ", stim_type)
        if stim_type in ["interleaved"]:
            n = 0
            abr_data, tb, stim = self.average_across_traces(
                fd, n, protocol, date, high_pass_filter=high_pass_filter
            )
            # print(len(frlist), len(dblist))
            abr_data = abr_data.reshape(len(dblist), len(frlist), -1)
            # print("calculated new tb")
        else:
            tb = np.linspace(0, len(abr_data[0, 0]) / self.fsamp, len(abr_data[0, 0]))
        metadata = {
            "type": "pyabr3",
            "filename": filename,
            "subject_id": subject_id,
            "age": age,
            "sex": sex,
            "amplifier_gain": 1e4,
            "strain": strain,
            "weight": weight,
            "genotype": genotype,
            "record_frequency": self.fsamp,
        }

        self.plot_abrs(
            abr_data=abr_data,
            tb=tb,
            stim_type=stim_type,
            scale=scale,
            dblist=dblist,
            frlist=frlist,
            metadata=metadata,
            maxdur=maxdur,
            highpass=300.0,
            use_matplotlib=True,
            pdf=pdf,
        )


def do_directory(
    directory_name: Union[Path, str],
    output_file: Union[Path, str],
    subject_prefix: str = "CBA",
    hide_treatment: bool = False,
):
    AR = AnalyzeABR()
    AR.set_hide_treatment(hide_treatment)
    ABR4 = read_abr4.READ_ABR4()
    maxdur = 12.0
    HPF = 300.0
    # base directory
    directory_name = "/Volumes/Pegasus_002/ManisLab_Data3/abr_data/Reggie_NIHL"
    subjs = Path(directory_name).glob(f"{subject_prefix}*")
    with PdfPages(output_file) as pdf:
        for subj in subjs:
            if not subj.is_dir():
                continue
            fns = subj.glob("*")
            for f in fns:
                fname = f.name.lower()
                if fname.startswith("click"):
                    print("Click file: ", f)
                    pfiles = list(Path(f).glob("*_click_*.p"))
                    if len(pfiles) > 0:
                        AR.read_and_average_abr_files(
                            filename=str(pfiles[0]), high_pass_filter=HPF, maxdur=maxdur, pdf=pdf
                        )
                    else:
                        ABR4.plot_dataset(
                            AR,
                            datatype="click",
                            subject=f.parent,
                            topdir=directory_name,
                            subdir=f.name,
                            highpass=HPF,
                            maxdur=maxdur,
                            pdf=pdf,
                        )
                elif fname.startswith("tone"):
                    print("Tone file: ", f)
                    ABR4.plot_dataset(
                        AR,
                        datatype="tone",
                        subject=f.parent,
                        topdir=directory_name,
                        subdir=f.name,
                        highpass=HPF,
                        maxdur=maxdur,
                        pdf=pdf,
                    )
                elif fname.lower().startswith("interleaved"):
                    print("Interleaved file: ", f)
                    files = list(Path(f).glob("*.p"))
                    print(f, "\n     # interleaved files: ", len(files))
                    # d = AR.read_abr_file(str(files[0]))
                    # print(d["stimuli"].keys())
                    HPF = 300.0
                    # AR.show_stimuli(files[0])

                    AR.read_and_average_abr_files(
                        filename=str(files[0]),
                        high_pass_filter=HPF,
                        maxdur=maxdur,
                        pdf=pdf,
                    )
                else:
                    if not f.name.startswith(".DS_Store"):
                        raise ValueError(f"File {f} for data file {fname:s} not recognized")


if __name__ == "__main__":

    do_directory(
        directory_name="/Volumes/Pegasus_002/ManisLab_Data3/abr_data/Reggie_NIHL",
        output_file="NIHL_VGAT-EYFP_ABRs_combined.pdf",
        subject_prefix="VGAT-EYFP",
        hide_treatment=True,
    )

    
