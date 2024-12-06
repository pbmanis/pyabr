
import tdt
import time
proj = tdt.DSPProject(interface="USB")
pa5_1 = tdt.util.connect_pa5(interface="USB", device_id=1)
pa5_2 = tdt.util.connect_pa5(interface="USB", device_id=1)
circuit = proj.load_circuit("tdt/abrs_v2.rcx", "RP2")
circuit.start(0.25)
proj.trigger('B', 'high')
time.sleep(1)
proj.trigger('B', 'low')

pa5_1.SetAtten(40)
time.sleep(1)
pa5_1.SetAtten(120)

