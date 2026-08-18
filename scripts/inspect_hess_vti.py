import os
import segyio
import time

def inspect_segy(filepath):
    print(f"\n======================================")
    print(f"Inspecting: {os.path.basename(filepath)}")
    print(f"======================================")
    
    t0 = time.time()
    try:
        with segyio.open(filepath, "r", ignore_geometry=True) as segyfile:
            t1 = time.time()
            print(f"File opened in {t1-t0:.2f} seconds.")
            
            total_traces = segyfile.tracecount
            print(f"Total Traces: {total_traces}")
            
            if total_traces == 0:
                return None
            
            nsamples = segyfile.samples.size
            dt_microseconds = segyfile.bin[segyio.BinField.Interval]
            dt = dt_microseconds / 1e6
            
            print(f"Time Samples per Trace: {nsamples}")
            print(f"Sampling Rate (dt): {dt} seconds")
            
            print("Parsing headers for geometry...")
            t0 = time.time()
            
            offsets = segyfile.attributes(segyio.TraceField.offset)[:]
            source_x = segyfile.attributes(segyio.TraceField.SourceX)[:]
            source_y = segyfile.attributes(segyio.TraceField.SourceY)[:]
            group_x = segyfile.attributes(segyio.TraceField.GroupX)[:]
            group_y = segyfile.attributes(segyio.TraceField.GroupY)[:]
            
            field_records = segyfile.attributes(segyio.TraceField.FieldRecord)[:]
            shots = set(field_records)
            
            t1 = time.time()
            print(f"Headers parsed in {t1-t0:.2f} seconds.")
            
            num_shots = len(shots)
            avg_traces_per_shot = total_traces / num_shots if num_shots > 0 else 0
            
            print(f"Number of Shot Gathers: {num_shots}")
            print(f"Average Traces per Shot: {avg_traces_per_shot:.1f}")
            
            min_offset, max_offset = offsets.min(), offsets.max()
            print(f"Offset Range: [{min_offset}, {max_offset}]")
            print(f"Source X Range: [{source_x.min()}, {source_x.max()}]")
            print(f"Source Y Range: [{source_y.min()}, {source_y.max()}]")
            print(f"Receiver X Range: [{group_x.min()}, {group_x.max()}]")
            print(f"Receiver Y Range: [{group_y.min()}, {group_y.max()}]")
            
            return {
                "total_traces": total_traces,
                "shots": num_shots,
                "traces_per_shot": avg_traces_per_shot,
                "nsamples": nsamples,
                "dt": dt,
                "offset_range": (min_offset, max_offset)
            }
    except Exception as e:
        print(f"Error reading {filepath}: {e}")
        return None

def main():
    data_dir = os.path.join(os.path.dirname(__file__), '../data/hess_vti')
    
    file_i = os.path.join(data_dir, 'timodel_shot_data_I.segy')
    file_ii_1 = os.path.join(data_dir, 'timodel_shot_data_II_shot001-320.segy')
    file_ii_2 = os.path.join(data_dir, 'timodel_shot_data_II_shot321-720.segy')
    
    info_i = None
    info_ii_1 = None
    info_ii_2 = None
    
    if os.path.exists(file_i):
        info_i = inspect_segy(file_i)
    else:
        print(f"Data I not found at {file_i}")
        
    if os.path.exists(file_ii_1):
        info_ii_1 = inspect_segy(file_ii_1)
    else:
        print(f"Data II (1) not found at {file_ii_1}")
        
    if os.path.exists(file_ii_2):
        info_ii_2 = inspect_segy(file_ii_2)
    else:
        print(f"Data II (2) not found at {file_ii_2}")
    
    print("\n======================================")
    print("Alignment Check (Data I vs Data II)")
    print("======================================")
    
    total_traces_ii = (info_ii_1['total_traces'] if info_ii_1 else 0) + (info_ii_2['total_traces'] if info_ii_2 else 0)
    print(f"Total Traces in Data I: {info_i['total_traces'] if info_i else 0}")
    print(f"Total Traces in Data II (combined): {total_traces_ii}")
    
    if info_i and (info_i['total_traces'] == total_traces_ii):
        print("-> PERFECT ALIGNMENT: 1:1 trace matching between Data I and Data II.")
    else:
        print("-> MISMATCH DETECTED: Traces do not align exactly.")

if __name__ == '__main__':
    main()
