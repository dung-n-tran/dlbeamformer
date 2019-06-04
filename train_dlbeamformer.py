import numpy as np
import matplotlib.pyplot as plt
import pickle

from scipy.signal import stft, istft, get_window
from scipy.fftpack import fft, fftshift, fftfreq
from tqdm import tnrange, tqdm_notebook, tqdm
from dlbeamformer_utilities import *
from dlbeamformers import *
from utilities import *
from IPython.display import Audio

random_seed = 0

def main(datapath, n_interferences=1):
    print("Run train dlbeamformers")    
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
    
    print("Number of interferences:", n_interferences)
    
    azimuth_step = 5
#     n_samples_each_config = len(train_data)
    n_samples_each_config = 100
    training_azimuths = list(np.arange(0, 360, azimuth_step))
    training_elevations = np.array([-90])
    training_angles = {
        "elevation": training_elevations,
        "azimuth": training_azimuths
    }

    training_tf_filename = \
        "circular_CMU_ARCTIC_tf_training_covariance_data_azimuth_step_{}_config_trainning_samples_{}_n_interferences_{}.pkl".format(
            azimuth_step, n_samples_each_config, n_interferences
        )

    training_tf_path = "/data/dung/dlbeamformer/tf_training_data"
    training_tf_filepath = os.path.join(training_tf_path, training_tf_filename)

    training_interference_covariance = prepare_multichannel_covariance_data(
                        array_geometry, train_data, n_interferences, training_angles,
                        azimuth_step, n_samples_each_config, sampling_frequency,
                        stft_params, training_tf_filepath, random_seed=0, SAVE=True)

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
            source_angles, stft_params, angle_grid, diagonal_loading_param=10, bf_type="NC")
    desired_null_width = 5 # [degree]
    null_constraint_threshold = 0.001
    n_atoms_each_config = 1
    batch_size = 900
    dlbeamformer.fit(training_interference_covariance, desired_null_width, 
                    null_constraint_threshold, eigenvalue_percentage_threshold=0.99,
                    batch_size=batch_size, n_atoms_each_config=n_atoms_each_config)
    dlbeamformer_filename = \
        "circular_CMU_ARCTIC_dlbeamformer_azimuth_step_{}_config_atoms_{}_n_interferences_{}_batch_size_{}.pkl".format(
            azimuth_step, n_atoms_each_config, n_interferences, batch_size)
    trained_models_path = "/data/dung/dlbeamformer/trained_models"
    dlbeamformer_filepath = os.path.join(trained_models_path, dlbeamformer_filename)

    with open(dlbeamformer_filepath, 'wb') as f:
            pickle.dump(dlbeamformer, f, pickle.HIGHEST_PROTOCOL)
            
    dlbeamformer_weights_filename = \
        "weights_circular_CMU_ARCTIC_dlbeamformer_azimuth_step_{}_config_atoms_{}_n_interferences_{}_batch_size_{}.pkl".format(
            azimuth_step, n_atoms_each_config, n_interferences, batch_size)
    weights_path = "/data/dung/dlbeamformer/weights"
    dlbeamformer_weights_filepath = os.path.join(weights_path, dlbeamformer_weights_filename)

    with open(dlbeamformer_weights_filepath, 'wb') as f:
            pickle.dump(dlbeamformer.weights_, f, pickle.HIGHEST_PROTOCOL)
            
if __name__=="__main__":
    datapath = "CMU_ARCTIC/cmu_us_bdl_arctic/wav"
    n_interferences = 2
    main(datapath, n_interferences)