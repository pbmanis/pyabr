from numpy import arange, sin, pi
from tdt import DSPProject, DSPError
import matplotlib.pyplot

def test_circuit():
    project = DSPProject(interface="USB")
    circuit = project.load_circuit('tdt/record_microphone.rcx', 'RP2')

    circuit.start()

    # Configure the data tags
    circuit.cset_tag('record_del_n', 25, 'ms', 'n')
    circuit.cset_tag('record_dur_n', 500, 'ms', 'n')

    # Compute and upload the waveform
    t = arange(0, 1, circuit.fs**-1)
    waveform = sin(2*pi*1e3*t)
    speaker_buffer = circuit.get_buffer('speaker', 'w')
    speaker_buffer.write(waveform)

    # Acquire the microphone data
    microphone_buffer = circuit.get_buffer('mic', 'r')
    data = microphone_buffer.acquire("A", 'running', False)
    print(data.shape)

if __name__ == "__main__":
    test_circuit()
    