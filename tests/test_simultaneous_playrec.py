import sounddevice as sd
import numpy as np
import pyqtgraph as pg
import time
import src.sound as sound


def playrec():
    fs = 44100
    duration = 5  # seconds

    sd.default.samplerate = fs
    sd.default.channels = 1
    sd.default.dtype = "float32"
    sd.default.device = [3, 4]
    sd.default.latency = "low"
    sd.default.blocksize = 1024
    sd.default.extra_settings = None
    sd.default.clip_off = False
    sd.default.dither_off = False
    sd.default.never_drop_input = False
    pip_start = [0.20] # + np.linspace(0, 0.05, 1)
    print(pip_start)
    sound_wave = sound.TonePip(
        rate=fs,
        duration=0.50,
        f0=4000.,
        dbspl=100.,
        pip_duration=0.005,
        ramp_duration=0.0005,
        pip_start= list(pip_start),
        alternate=False
    )

    wave = sound_wave.sound
    app = pg.mkQApp("Calibration Data Plot")
    win = pg.GraphicsLayoutWidget(show=True, title="ABR Data Plot")
    win.resize(500, 500)
    win.setWindowTitle(f"Testing Simultaneious")
  
    pl = win.addPlot(title=f"Output")
    win.nextRow()
    pl2 = win.addPlot(title=f"Input")

    n_reps = 10
    for i in range(n_reps):
        myrecording = sd.playrec(data=wave, samplerate=fs, dtype="float32", blocking=True)
        t = np.linspace(0, len(wave) / fs, len(wave))
        t_rec = np.linspace(0, len(myrecording) / fs, len(myrecording))
        pl.plot(t, wave)
        pl2.plot(t_rec, myrecording.squeeze(), pen=pg.mkPen(i, n_reps, width=2))
    pg.exec()


if __name__ == "__main__":
    print(sd.query_devices())
    playrec()
