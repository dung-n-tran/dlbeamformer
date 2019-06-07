import numpy as np
import matplotlib.pyplot as plt
import pickle

from scipy.signal import stft, istft, get_window
from scipy.fftpack import fft, fftshift, fftfreq
from IPython.display import Audio
from tqdm import tnrange, tqdm_notebook, tqdm
from dlbeamformer_utilities import *
from dlbeamformers import *
from utilities import *
from IPython.display import Audio

random_seed = 0
# Make pretty figures
palette, cmap = config_figures()

VISUALIZE_BEAMPATTERNS = True
plt.style.use("seaborn-poster")
main_color = (59/255, 157/255, 180/255)

datapath = "CMU_ARCTIC/cmu_us_bdl_arctic/wav"
train_data, test_data = load_data(datapath)

sampling_frequency, stft_params, sound_speed = parse_parameters()

signal_max_frequency = sampling_frequency / 2

# Array geometry
pos_x = np.array([-35.0, -35.0, 0.0, 35.0, 35.0, 0.0, 0.0]) * 1e-3
pos_y = np.array([20.0, -20.0, -40.0, -20.0, 20.0, 40.0, 0.0]) * 1e-3
n_mics = len(pos_x)
pos_z = np.zeros(n_mics)
array_geometry = np.row_stack((pos_x, pos_y, pos_z))

# Fix elevation angle
elevation = -90 # [degree]

# Source/Target/Look angles
elevation_s = np.array([elevation]) # [degree]
azimuth_s = np.array([180])
source_steering_vectors = compute_steering_vectors(array_geometry, 
        sampling_frequency=sampling_frequency, n_fft_bins=stft_params["n_fft_bins"], 
        elevation_grid=elevation_s, 
        azimuth_grid=azimuth_s)

# Scanning angles
scanning_elevation_grid = np.array([elevation]) # [degree]
scanning_azimuth_grid = np.arange(0, 360, 0.1) # [degree]
scanning_steering_vectors = compute_steering_vectors(array_geometry, 
        sampling_frequency=sampling_frequency, n_fft_bins=stft_params["n_fft_bins"], 
        elevation_grid=scanning_elevation_grid, 
        azimuth_grid=scanning_azimuth_grid)

##### TRAIN DICTIONARY
SAVE_DICT = False
np.random.seed(random_seed)
n_interferences = 1
azimuth_step = 5
n_samples_each_config = len(train_data)
training_azimuths = list(np.arange(0, 360, azimuth_step))
training_elevations = np.array([-90])
training_angles = {
    "elevation": training_elevations,
    "azimuth": training_azimuths
}
noisy = False
training_tf_filename = \
    "circular_CMU_ARCTIC_tf_training_covariance_data_azimuth_step_{}_config_trainning_samples_{}_n_interferences_{}.pkl".format(
        azimuth_step, n_samples_each_config, n_interferences
    )
if noisy == True:
    training_tf_filename = "noisy_" + training_tf_filename
training_tf_path = "/data/dung/dlbeamformer/tf_training_data"
training_tf_filepath = os.path.join(training_tf_path, training_tf_filename)

training_interference_covariance = prepare_multichannel_covariance_data(
                    array_geometry, train_data, n_interferences, training_angles,
                    azimuth_step, n_samples_each_config, sampling_frequency,
                    stft_params, training_tf_filepath, random_seed=0, SAVE=False)

source_angles = [
    {
        "elevation": elevation_s,
        "azimuth": azimuth_s
    }
]
angle_grid = {
    "elevation": scanning_elevation_grid,
    "azimuth": scanning_azimuth_grid
}
dlbeamformer = DictionaryLearningBeamformer(array_geometry, sampling_frequency,
        source_angles, stft_params, angle_grid, diagonal_loading_param=5, bf_type="NC")
desired_null_width = 5 # [degree]
null_constraint_threshold = 0.001
n_atoms_each_config = 1
batch_size = 100
dlbeamformer.fit(training_interference_covariance, desired_null_width, 
                null_constraint_threshold, eigenvalue_percentage_threshold=0.99,
                batch_size=batch_size, n_atoms_each_config=n_atoms_each_config)
dlbeamformer_filename = \
    "circular_CMU_ARCTIC_dlbeamformer_azimuth_step_{}_config_atoms_{}_n_interferences_{}_batch_size_{}.pkl".format(
        azimuth_step, n_atoms_each_config, n_interferences, batch_size)
trained_models_path = "/data/dung/dlbeamformer/trained_models"
dlbeamformer_filepath = os.path.join(trained_models_path, dlbeamformer_filename)

if SAVE_DICT:
    filepath = os.path.join(folder, filename + ".pkl")
    with open(dlbeamformer_filepath, 'wb') as handle:
        pickle.dump(dlbeamformer, handle, protocol=pickle.HIGHEST_PROTOCOL)
        
#####################################################
# Run Monte Carlo simulation to compute SINR curves #
#####################################################
np.random.seed(random_seed)
n_interferences = 1
input_sinrs_db = np.arange(-15, 16, 2)
input_inr_db = 0
input_inr = from_db(input_inr_db)
beamformer_list = ["DL-MPDR", "DL-MVDR", "DS", "MVDR", "MPDR"]
# beamformer_list = ["mvdr"]
beamformers = {}
for beamformer_name in beamformer_list:
    beamformers[beamformer_name] = {
        "weights": [None]*len(input_sinrs_db),
        "sinr_db": [None]*len(input_sinrs_db),
        "average_sinr_db": [],
        "out": [None]*len(input_sinrs_db),
        "ir_db": [None] * len(input_sinrs_db),
        "average_ir_db": [],
        "nr_db": [None] * len(input_sinrs_db),
        "average_nr_db": [],
        "sr_db": [None] * len(input_sinrs_db),
        "average_sr_db": []
    }
        
n_MC_iters = 1
for i_input_sinr in tqdm(range(len(input_sinrs_db)), desc="Input SINR"):
    input_sinr = from_db(input_sinrs_db[i_input_sinr])
    for beamformer_name in beamformer_list:
        beamformers[beamformer_name]["weights"][i_input_sinr] = []
        beamformers[beamformer_name]["sinr_db"][i_input_sinr] = []
        beamformers[beamformer_name]["out"][i_input_sinr] = []
        beamformers[beamformer_name]["ir_db"][i_input_sinr] = []
        beamformers[beamformer_name]["nr_db"][i_input_sinr] = []
        beamformers[beamformer_name]["sr_db"][i_input_sinr] = []
    for i_MC_iter in tqdm(range(n_MC_iters), desc="Monte Carlo iterations"):
    
        source = {
            "signal": test_data[np.random.choice(len(test_data))],
            "elevation": elevation_s,
            "azimuth": azimuth_s
        }
        
        interferences = []
        interference_signals = []
        for i_interference in range(n_interferences):
            random_azimuth_1 = np.random.uniform(scanning_azimuth_grid[0], azimuth_s - 30)[0]
            random_azimuth_2 = np.random.uniform(azimuth_s + 30, scanning_azimuth_grid[-1])[0]
            random_azimuth = np.random.choice(np.array([random_azimuth_1, random_azimuth_2]))
            interference_signal = test_data[np.random.choice(len(test_data))]
            interference = {
                "signal": interference_signal,
                "elevation": np.array([elevation]),
                "azimuth": np.array([np.random.uniform(
                    scanning_azimuth_grid[0], scanning_azimuth_grid[-1])])
#                 "azimuth": np.array([30])
#                 "azimuth": np.array([random_azimuth])
            }
            interferences.append(interference)
        
        received_stft_multichannel, source_stft_multichannel, \
        interference_stft_multichannel, noise_stft_multichannel \
            = simulate_multichannel_tf_mixtures(array_geometry, source,
                interferences, sampling_frequency, stft_params, input_inr, input_sinr)
        
        for beamformer_name in beamformer_list:        
            if beamformer_name.lower() == "ds":
                tf_frames_multichannel = None
            elif beamformer_name.lower() in ["mvdr", "dl-mvdr"]:
                tf_frames_multichannel = interference_stft_multichannel
            elif beamformer_name.lower() in ["mpdr", "dl-mpdr"]:
                tf_frames_multichannel = received_stft_multichannel

            if beamformer_name.lower() in ["dl-mvdr", "dl-mpdr"]:
                source_angle_index = 0
                tf_beamformer = dlbeamformer.choose_weights(source_angle_index,
                                    tf_frames_multichannel)
            else:
                tf_beamformer = compute_tf_beamformers(source_steering_vectors[:, 0, 0, :], 
                        beamformer_name=beamformer_name,
                        tf_frames_multichannel=tf_frames_multichannel,
                        diagonal_loading_param=5)

            # Compute beamformer output and SINR
            tf_out, out, _ = compute_tf_beamformer_output(tf_beamformer, 
                                received_stft_multichannel, sampling_frequency, 
                                stft_params)
            
            sinr_db, sinr  = compute_sinr(source_stft_multichannel, 
                interference_stft_multichannel+noise_stft_multichannel, tf_beamformer)

            # Compute interference output and interference reduction
            interference_tf_out, interference_out, _ = compute_tf_beamformer_output(tf_beamformer, 
                                interference_stft_multichannel, sampling_frequency, 
                                stft_params)
            
            interference_reduction = compute_power(interference_stft_multichannel[:, 0, :]) / compute_power(interference_tf_out)
            interference_reduction_db = to_db(interference_reduction)
            
            # Compute noise output and noise reduction
            noise_tf_out, noise_out, _ = compute_tf_beamformer_output(tf_beamformer, 
                                noise_stft_multichannel, sampling_frequency, 
                                stft_params)
            
            noise_reduction = compute_power(noise_stft_multichannel[:, 0, :]) / compute_power(noise_tf_out)
            noise_reduction_db = to_db(noise_reduction)
            
            # Compute source output and source distortion/reduction
            source_tf_out, source_out, _ = compute_tf_beamformer_output(tf_beamformer, 
                                source_stft_multichannel, sampling_frequency, 
                                stft_params)
            
            source_reduction = compute_power(source_stft_multichannel[:, 0, :]) / compute_power(source_tf_out)
            source_reduction_db = to_db(source_reduction)
            
            beamformers[beamformer_name]["weights"][i_input_sinr].append(tf_beamformer)
            beamformers[beamformer_name]["sinr_db"][i_input_sinr].append(sinr_db[0][0])
            beamformers[beamformer_name]["out"][i_input_sinr].append(out)
            beamformers[beamformer_name]["ir_db"][i_input_sinr].append(interference_reduction_db)
            beamformers[beamformer_name]["nr_db"][i_input_sinr].append(noise_reduction_db)
            beamformers[beamformer_name]["sr_db"][i_input_sinr].append(source_reduction_db)

    for beamformer_name in beamformers.keys():
        beamformers[beamformer_name]["average_sinr_db"].append(
            to_db(np.mean(from_db(np.asarray(beamformers[beamformer_name]["sinr_db"][i_input_sinr])))))
        beamformers[beamformer_name]["average_ir_db"].append(
            to_db(np.mean(from_db(np.asarray(beamformers[beamformer_name]["ir_db"][i_input_sinr])))))
        beamformers[beamformer_name]["average_nr_db"].append(
            to_db(np.mean(from_db(np.asarray(beamformers[beamformer_name]["nr_db"][i_input_sinr])))))
        beamformers[beamformer_name]["average_sr_db"].append(
            to_db(np.mean(from_db(np.asarray(beamformers[beamformer_name]["sr_db"][i_input_sinr])))))
        
# Save SINR outputs. Need to modify folder name accordingly
folder = "/data/dung/dlbeamformer/out"
filename = "beamformers_input_inr_{}dB".format(input_inr_db)
filepath = os.path.join(folder, filename + ".pkl")
# with open(filepath, 'wb') as handle:
#     pickle.dump(beamformers, handle, protocol=pickle.HIGHEST_PROTOCOL)        

##### Visualize SINR curves for a given input INR
import matplotlib as mpl
mpl.rcParams['grid.linewidth'] = 0.3
mpl.rcParams['grid.alpha'] = 0.3
folder = "figures"
filename = "sinr_input_inr_{}dB".format(input_inr_db)
filepath = os.path.join(folder, filename + ".jpg")
figsize = (9, 6)
fig = plt.figure(figsize=figsize); ax = fig.add_subplot(111);
for i_beamformer, beamformer_name in enumerate(beamformer_list):
    ax.plot(input_sinrs_db, beamformers[beamformer_name]["average_sinr_db"],
                label=beamformer_name, color=palette[i_beamformer])
    y_pos = beamformers[beamformer_name]["average_sinr_db"][-1] - 0.5 
    if beamformer_name == "DL-MVDR":
        y_pos += 0
    if beamformer_name == "MVDR":
        y_pos += 1.2
    ax.text(input_sinrs_db[-1]+0.5, y_pos, beamformer_name, fontsize=12, color=palette[i_beamformer])
ax.fill_between(input_sinrs_db, beamformers["DL-MPDR"]["average_sinr_db"], 
                beamformers["MPDR"]["average_sinr_db"], color=palette[0], alpha=0.2)
ax.set_xlim(input_sinrs_db[0], input_sinrs_db[-1]);
ax.set_xlabel("Input SINR"); ax.set_ylabel("Output SINR [dB]");
ax.spines["top"].set_alpha(0.3)
ax.spines["bottom"].set_alpha(0.3)
ax.spines["right"].set_alpha(0.3)
ax.spines["left"].set_alpha(.3)
fig.savefig(filepath, dpi=600)

filename = "sr_input_inr_{}dB".format(input_inr_db)
filepath = os.path.join(folder, filename + ".jpg")
fig = plt.figure(figsize=figsize); ax = fig.add_subplot(111);
for i_beamformer, beamformer_name in enumerate(beamformer_list):
    ax.plot(input_sinrs_db, beamformers[beamformer_name]["average_sr_db"],
                label=beamformer_name, color=palette[i_beamformer])
    y_pos = beamformers[beamformer_name]["average_sr_db"][-1] - 0.5 
#     ax.text(input_sinrs_db[-1]+0.5, y_pos, beamformer_name, fontsize=16, color=palette[i_beamformer])
ax.set_xlim(input_sinrs_db[0], input_sinrs_db[-1]); ax.set_ylim(-20, 20)
ax.set_xlabel("Input SINR"); ax.set_ylabel("Output SR [dB]");
ax.spines["top"].set_alpha(0.3)
ax.spines["bottom"].set_alpha(0.3)
ax.spines["right"].set_alpha(0.3)
ax.spines["left"].set_alpha(.3)
legend = ax.legend();
for text in legend.get_texts():
    text.set_color("gray")
fig.savefig(filepath, dpi=600)

filename = "ir_input_inr_{}dB".format(input_inr_db)
filepath = os.path.join(folder, filename + ".jpg")
fig = plt.figure(figsize=figsize); ax = fig.add_subplot(111);
for i_beamformer, beamformer_name in enumerate(beamformer_list):
    ax.plot(input_sinrs_db, beamformers[beamformer_name]["average_ir_db"],
                label=beamformer_name, color=palette[i_beamformer])
    y_pos = beamformers[beamformer_name]["average_ir_db"][-1] - 0.5 
    ax.text(input_sinrs_db[-1]+0.5, y_pos, beamformer_name, fontsize=12, color=palette[i_beamformer])    
ax.fill_between(input_sinrs_db, beamformers["DL-MPDR"]["average_ir_db"], 
                beamformers["MPDR"]["average_ir_db"], color=palette[0], alpha=0.2)    
ax.set_xlim(input_sinrs_db[0], input_sinrs_db[-1]);
ax.set_xlabel("Input SINR"); ax.set_ylabel("Output IR [dB]");
ax.spines["top"].set_alpha(0.3)
ax.spines["bottom"].set_alpha(0.3)
ax.spines["right"].set_alpha(0.3)
ax.spines["left"].set_alpha(.3)
fig.savefig(filepath, dpi=600)

filename = "nr_input_inr_{}dB".format(input_inr_db)
filepath = os.path.join(folder, filename + ".jpg")
fig = plt.figure(figsize=figsize); ax = fig.add_subplot(111);
for i_beamformer, beamformer_name in enumerate(beamformer_list):
    ax.plot(input_sinrs_db, beamformers[beamformer_name]["average_nr_db"],
                label=beamformer_name, color=palette[i_beamformer])
    y_pos = beamformers[beamformer_name]["average_nr_db"][-1] - 0.5
    if beamformer_name == "DL-MVDR":
        y_pos += 1.2
    ax.text(input_sinrs_db[-1]+0.5, y_pos, beamformer_name, fontsize=12, color=palette[i_beamformer])    
ax.fill_between(input_sinrs_db, beamformers["DL-MPDR"]["average_nr_db"], 
                beamformers["MPDR"]["average_nr_db"], color=palette[0], alpha=0.2)    
ax.set_xlim(input_sinrs_db[0], input_sinrs_db[-1]);
ax.set_xlabel("Input SINR"); ax.set_ylabel("Output NR [dB]");
ax.spines["top"].set_alpha(0.3)
ax.spines["bottom"].set_alpha(0.3)
ax.spines["right"].set_alpha(0.3)
ax.spines["left"].set_alpha(.3)
fig.savefig(filepath, dpi=600)