import os
from os import listdir
from os.path import join
from scipy.io import wavfile
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from IPython.display import Audio, display
from scipy.signal import stft, istft

def load_data(datapath):
    train_data_folder = join(datapath, 'train')
    test_data_folder = join(datapath, 'test')
    train_data = []
    test_data = []
    train_data_filenames = [f for f in listdir(train_data_folder) if os.path.isfile( join(train_data_folder, f))]
    test_data_filenames = [f for f in listdir(test_data_folder) if os.path.isfile( join(test_data_folder, f))]

    for i_train_data_filename in range(len(train_data_filenames)):
        f_path = join(train_data_folder, train_data_filenames[i_train_data_filename])
        if f_path.endswith('.wav'):
            sampling_frequency, train_data_example = wavfile.read(f_path)
        train_data.append(train_data_example)

    min_test_length = np.inf
    for i_test_data_filename in range(len(test_data_filenames)):
        f_path = join(test_data_folder, test_data_filenames[i_test_data_filename])
        if f_path.endswith('.wav'):
            sampling_frequency, test_data_example = wavfile.read(f_path)
        test_example_length = len(test_data_example)
        if test_example_length > 32000:
            if test_example_length < min_test_length:
                min_test_length = test_example_length
            test_data.append(test_data_example)
    for i_test in range(len(test_data)):
        test_data[i_test] = test_data[i_test][0:min_test_length]

    return train_data, test_data

def parse_parameters():
    from configparser import ConfigParser
    from scipy.signal import get_window
    config = ConfigParser()
    config.read('config.INI');
    params = config['PARAMS']
    sampling_frequency = int(params['sampling_frequency'])
    n_samples_per_frame = int(params['n_samples_per_frame'])
    n_fft_bins = (int) (n_samples_per_frame / 2) 
    hop_size = (int) (n_samples_per_frame / 2)
    stft_window_name = params['stft_window_name']
    stft_window = get_window("hann", n_samples_per_frame)
    stft_params = {
        "n_samples_per_frame": n_samples_per_frame,
        "n_fft_bins": n_fft_bins,
        "hop_size": hop_size,
        "window": stft_window
    }
    sound_speed = int(config["CONSTANTS"]["SOUND_SPEED"])
    return sampling_frequency, stft_params, sound_speed

def config_figures():
    import matplotlib as mpl
    mpl.rcParams['axes.spines.top'] = False
    mpl.rcParams['axes.spines.right'] = False
    mpl.rcParams['axes.grid'] = True
    mpl.rcParams['grid.color'] = 'gray'
    mpl.rcParams['grid.linewidth'] = 0.1
    mpl.rcParams['grid.alpha'] = 0.1
    mpl.rcParams['xtick.color'] = 'gray'; mpl.rcParams['ytick.color'] = 'gray'
    mpl.rcParams['axes.labelcolor'] = 'gray';
    mpl.style.use('seaborn-poster')
    
    from seaborn import set_palette
    palette = ["#3b9db4", "#9b59b6", "#95a5a6", "#2ecc71", "#34495e", "#3498db", "#e74c3c"]
    set_palette(palette)
    
    cmap = "RdBu_r"
    
    return palette, cmap

def to_db(x):
    return 10*np.log10(np.abs(x))

def from_db(x):
    return 10**(x/10)

def compute_power(x):
    return np.sum( np.multiply( np.abs(x), np.abs(x) ) )

def visualize_tf(tf_frames, sampling_frequency, figsize=(9, 6), cmap="coolwarm"):
    fig = plt.figure(figsize=figsize); ax = fig.add_subplot(111)
    ax.imshow(to_db(tf_frames), origin='lower', aspect='auto', cmap=cmap,
          extent=[0, tf_frames.shape[1], 0, sampling_frequency/2 * 1e-3])
    ax.set_xlabel("Time frames"); ax.set_ylabel("Frequency [KHz]");    
    return ax

def play_tf_frames(tf_frames, sampling_frequency, stft_params):
    stft_window = stft_params["window"]
    n_samples_per_frame = stft_params["n_samples_per_frame"]
    hop_size = stft_params["hop_size"]
    t, ss = istft(tf_frames, fs=sampling_frequency, 
        window=stft_window, nperseg=n_samples_per_frame, 
        noverlap=n_samples_per_frame-hop_size,
        nfft=n_samples_per_frame, boundary=True)
    display(Audio(ss, rate=sampling_frequency, autoplay=True))