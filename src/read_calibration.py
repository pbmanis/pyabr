import numpy as np
from pathlib import Path
import scipy.io
from pyqtgraph import configfile
import matplotlib.pyplot as mpl
#


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

    return caldata

def plot_calibration(caldata):
    f, ax = mpl.subplots(1,1)
    freqs = caldata['freqs']
    ax.semilogx(freqs, caldata['maxdb'], 'ro-')
    ax.semilogx(freqs, caldata['db_cs'], 'k--')
    ax.semilogx(freqs, caldata['db_bp'], 'g--')
    ax.semilogx(freqs, caldata['db_nf'], 'b-')
    ax.set_xlabel ("F, Hz")
    ax.set_ylabel("dB SPL")
    txt = f"Gain: {caldata['gain']:.1f}  Cal attn: {caldata['calattn']:.1f} dB, "
    txt += f"Speaker: {caldata['spkr']:s}, Mic: {caldata['mic']:s}, date: {caldata['date']:s}\nFile: {str(fn):s}"
    mpl.suptitle(txt, fontsize=7)
    ax.grid(True, which="both")
    f.tight_layout()
    mpl.show()

if __name__ == "__main__":
    # get the latest calibration file:
    cfg = configfile.readConfigFile("config/abrs.cfg")
    print(cfg)
    fn = Path(cfg['calfile'])
    # fn = Path("E:/Users/Experimenters/Desktop/ABR_Code/frequency_MF1.cal")
    print(fn.is_file())

    d = get_calibration_data(fn)
    plot_calibration(d)