import pickle
import matplotlib.pyplot as mpl
def read_abr(fn):
    with open(fn, "rb") as fh:
        d = pickle.load(fh)
    print(d.keys())
    return d

if __name__ == "__main__":
    fn = "abr_data/clicks/2024-11-13_click_000_001.p"
    d = read_abr(fn)
    print(d['data'].shape)
    mpl.plot(d['data'])
    mpl.show()
    print(d['calibration'])


