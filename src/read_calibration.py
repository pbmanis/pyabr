import numpy as np
from pathlib import Path
import scipy.io
from pyqtgraph import configfile
import matplotlib.pyplot as mpl
import pyqtgraph as pg
#

use_matplotlib = False

"""File structure is: 
CAL.RefSPL = CALIBRATION.SPLCAL.maxtones;
CAL.Freqs = spkr_freq; [1]
CAL.maxdB = maxdB; [2]
CAL.dBSPL = dbspl_cs; [3]
CAL.dBSPL_bp = dbspl_bp;
CAL.dBSPL_nf = dbspl_nf; [4]
CAL.Vmeas = Vrms_cs; [5]
CAL.Vmeas_bp = Vrms_bp; [6]
CAL.Gain = 20; % dB setting microphone amplifier gain
CAL.CalAttn = CALIBRATION.SPKR.CalAttn; % attenuator setting at which calibration was done
CAL.Speaker = Speaker;
CAL.Microphone = Mic;
CAL.Date = date;
CAL.DateTime = datetime();

cs refers to cosinor measurements.
bp refers to bandpass (narrow band filter)
nf refers to noise floor.
maxdB refers to max sound pressure level at 0 dB attenuation.

"""

def get_calibration_data(fn):

    dm = scipy.io.loadmat(fn, appendmat=False,  squeeze_me=True)
    d = dm["CAL"].item()
    caldata = {}
    caldata['refspl'] = d[0]
    # print("Ref SPL: ", caldata['maxspl'])
    caldata['freqs'] = d[1]
    caldata['maxdb'] = d[2]
    caldata['db_cs'] = d[3]
    caldata['db_bp'] = d[4]
    caldata['db_nf'] = d[5]
    caldata['vm_cs'] = d[6]
    caldata['vm_bp'] = d[7]
    caldata['gain'] = d[8]
    caldata['calattn'] = d[9]
    caldata['spkr'] = d[10]
    caldata['mic'] = d[11]
    caldata['date'] = d[12]
    caldata['filename'] = fn

    return caldata

def plot_calibration(caldata, plot_target = None):
    txt = f"Gain: {caldata['gain']:.1f}  Cal attn: {caldata['calattn']:.1f} dB, "
    txt += f"Speaker: {caldata['spkr']:s}, Mic: {caldata['mic']:s}, date: {caldata['date']:s}\nFile: {str(caldata['filename']):s}"
    if use_matplotlib:
        f, ax = mpl.subplots(1,1)
        freqs = caldata['freqs']
        ax.semilogx(freqs, caldata['maxdb'], 'ro-')
        ax.semilogx(freqs, caldata['db_cs'], 'k--')
        ax.semilogx(freqs, caldata['db_bp'], 'g--')
        ax.semilogx(freqs, caldata['db_nf'], 'b-')
        ax.set_xlabel ("F, Hz")
        ax.set_ylabel("dB SPL")
        fn = caldata['filename']
        txt = f"Gain: {caldata['gain']:.1f}  Cal attn: {caldata['calattn']:.1f} dB, "
        txt += f"Speaker: {caldata['spkr']:s}, Mic: {caldata['mic']:s}, date: {caldata['date']:s}\nFile: {str(fn):s}"
        mpl.suptitle(txt, fontsize=7)
        ax.grid(True, which="both")
        f.tight_layout()
        mpl.show()
    else:
        if plot_target is None:
            app = pg.mkQApp("Calibration Data Plot")
            win = pg.GraphicsLayoutWidget(show=True, title="ABR Data Plot")
            win.resize(500, 500)
            win.setWindowTitle(f"File: {caldata['filename']}")

        
            pl = win.addPlot(title=f"Calibration")
        else:
            pl = plot_target
        freqs = caldata['freqs']
        pl.setLogMode(x=True, y=False)
        pl.plot(freqs, caldata['maxdb'], pen='r')
        pl.plot(freqs, caldata['db_cs'], pen='w')
        pl.plot(freqs, caldata['db_bp'], pen='g')
        pl.plot(freqs, caldata['db_nf'], pen='b')
        # pl.setLogMode(x=True, y=False)
        pl.setLabel("bottom", "Frequency", units="Hz")
        pl.setLabel("left", "dB SPL")
        pl.showGrid(x=True, y=True)
        text_label = pg.LabelItem(txt, size="8pt", color=(255, 255, 255))
        text_label.setParentItem(pl)
        text_label.anchor(itemPos=(0.5, 0.05), parentPos=(0.5, 0.05))

        if plot_target is None:
            pg.exec()
if __name__ == "__main__":
    import sys
    configtype = "lab"
    cmd = sys.argv[1:]
    print("cmd: ", cmd)

    if cmd == "test":
        configfilename = "config/abrs_test.cfg"
    else:
        configtype = "lab"
        configfilename = "config/abrs.cfg"
    assert configtype in ["test", "lab"]
    # get the latest calibration file:
    cfg = configfile.readConfigFile(configfilename)
    print(cfg)
    fn = Path(cfg['calfile'])
    # fn = Path("E:/Users/Experimenters/Desktop/ABR_Code/frequency_MF1.cal")
    print(fn.is_file())

    d = get_calibration_data(fn)
    plot_calibration(d)