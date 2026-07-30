import numpy as np
import obspy
from obspy.core import Trace, Stream
import os

# Create directories
os.makedirs('data/labels', exist_ok=True)
os.makedirs('data/data', exist_ok=True)

ntraces = 64
nsamples = 128
num_samples_to_generate = 100

def ricker_wavelet(length=11, f0=25.0, dt=0.004):
    t = np.arange(-(length//2), length//2 + 1) * dt
    y = (1.0 - 2.0 * (np.pi**2) * (f0**2) * (t**2)) * np.exp(-(np.pi**2) * (f0**2) * (t**2))
    return y.astype(np.float32)

wavelet = ricker_wavelet()
half_w = len(wavelet) // 2

for i in range(num_samples_to_generate):
    # 1. TARGET (Labels): 1 or 2 hyperbolic diffractions with low amplitude
    label_data = np.zeros((ntraces, nsamples), dtype=np.float32)
    
    num_hyperbolas = np.random.choice([1, 2])
    for _ in range(num_hyperbolas):
        apex_trace = np.random.randint(10, 54)
        apex_time = np.random.randint(20, 90)
        velocity = np.random.uniform(0.6, 1.2)
        amplitude = np.random.uniform(0.2, 0.3)
        
        for tr in range(ntraces):
            dist = tr - apex_trace
            t_idx_exact = np.sqrt(apex_time**2 + (dist / velocity)**2)
            t_idx = int(np.round(t_idx_exact))
            
            # Inject wavelet
            if half_w <= t_idx < nsamples - half_w:
                label_data[tr, t_idx-half_w : t_idx+half_w+1] += wavelet * amplitude

    # 2. INPUT (Data): Same matrix + high-amplitude linear events + Gaussian noise
    input_data = label_data.copy()
    
    num_linear = np.random.randint(2, 5)
    for _ in range(num_linear):
        start_time = np.random.randint(10, 100)
        slope = np.random.uniform(-0.8, 0.8)
        linear_amplitude = np.random.uniform(0.8, 1.2) # High amplitude
        
        for tr in range(ntraces):
            t_idx_exact = start_time + slope * tr
            t_idx = int(np.round(t_idx_exact))
            
            if half_w <= t_idx < nsamples - half_w:
                input_data[tr, t_idx-half_w : t_idx+half_w+1] += wavelet * linear_amplitude

    # Add statistical Gaussian noise
    noise = np.random.normal(0, 0.03, (ntraces, nsamples)).astype(np.float32)
    input_data += noise

    # Convert to obspy.Stream (SU expects traces as items in a Stream)
    # The default obspy SU writer requires data to be float32 or float16.
    stream_label = Stream()
    stream_data = Stream()
    
    for tr in range(ntraces):
        trace_label = Trace(data=label_data[tr].astype(np.float32))
        trace_label.stats.delta = 0.004
        stream_label.append(trace_label)
        
        trace_data = Trace(data=input_data[tr].astype(np.float32))
        trace_data.stats.delta = 0.004
        stream_data.append(trace_data)

    # Save pairs
    stream_label.write(f'data/labels/{i}.su', format='SU')
    stream_data.write(f'data/data/{i}.su', format='SU')

print("Geração sintética concluída! 100 pares gerados em data/labels/ e data/data/")
