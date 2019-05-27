import os
from os import listdir
from os.path import join
from scipy.io import wavfile
import numpy as np
import matplotlib.pyplot as plt

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
    import matplotlib as plt
    plt.rcParams['axes.spines.top'] = False
    plt.rcParams['axes.spines.right'] = False
    plt.rcParams['axes.grid'] = True
    plt.rcParams['grid.color'] = 'gray'
    plt.rcParams['grid.linewidth'] = 0.25
    plt.rcParams['grid.alpha'] = 0.2
    plt.style.use('seaborn-talk')
    
    from seaborn import set_palette
    palette = ["#3498db", "#9b59b6", "#95a5a6", "#e74c3c", "#34495e", "#2ecc71"]
    set_palette(palette)
    
    cmap = "RdBu_r"
    
    return palette, cmap

def to_db(x):
    return 10*np.log10(np.abs(x))

def visualize_tf(tf_frames, sampling_frequency, figsize=(9, 6), cmap="coolwarm"):
    fig = plt.figure(figsize=figsize); ax = fig.add_subplot(111)
    ax.imshow(to_db(tf_frames), origin='lower', aspect='auto', cmap=cmap,
          extent=[0, tf_frames.shape[1], 0, sampling_frequency/2 * 1e-3])
    ax.set_xlabel("Time frames"); ax.set_ylabel("Frequency [KHz]");    
    return ax