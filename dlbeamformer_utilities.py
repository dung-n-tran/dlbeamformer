import numpy as np
from scipy.signal import stft
import matplotlib.pyplot as plt
from utilities import to_db
SOUND_SPEED = 340 # [m/s]
from scipy.signal import stft, istft
from tqdm import tqdm
import pickle

# Steering vectors
def compute_steering_vectors_single_frequency(array_geometry, frequency, elevation_grid, azimuth_grid):
    # wave number
    k = 2*np.pi*frequency/SOUND_SPEED

    n_mics = len(array_geometry[0])
    elevation_grid = elevation_grid * np.pi/180 # [degree] to [radian]
    azimuth_grid = azimuth_grid * np.pi/180 # [degree] to [radian]
    
    u = np.sin(elevation_grid.reshape(-1, 1)).dot(np.cos(azimuth_grid).reshape(1, -1))
    v = np.sin(elevation_grid.reshape(-1, 1)).dot(np.sin(azimuth_grid).reshape(1, -1))
    w = np.tile(np.cos(elevation_grid.reshape(-1, 1)), (1, azimuth_grid.shape[0]))

    x = u.reshape(u.shape[0], u.shape[1], 1)*array_geometry[0].reshape(1, 1, n_mics)
    y = v.reshape(v.shape[0], v.shape[1], 1)*array_geometry[1].reshape(1, 1, n_mics)
    z = w.reshape(w.shape[0], w.shape[1], 1)*array_geometry[2].reshape(1, 1, n_mics)

    return np.exp( -1j*k*(x + y + z))

def compute_steering_vectors(array_geometry, sampling_frequency, n_fft_bins, elevation_grid, azimuth_grid):
    n_elevations = len(elevation_grid)
    n_azimuths = len(azimuth_grid)
    n_mics = len(array_geometry[0])
    steering_vectors = np.zeros((n_fft_bins, n_elevations, n_azimuths, n_mics), dtype=np.complex64)
    for i_fft in range(n_fft_bins):
        frequency = (i_fft / n_fft_bins) * (sampling_frequency/2)
        steering_vectors[i_fft] = compute_steering_vectors_single_frequency(array_geometry, frequency, elevation_grid, azimuth_grid)
        
    return steering_vectors

def compute_tf_beampattern(beamformers, scanning_steering_vectors):
    n_fft_bins, n_elevations, n_azimuths, _ = scanning_steering_vectors.shape
    beampattern = np.zeros((n_fft_bins, n_elevations, n_azimuths), dtype=np.complex64)
    for i_fft_bin in range(n_fft_bins):
        for i_elevation in range(n_elevations):
            for i_azimuth in range(n_azimuths):
                v = scanning_steering_vectors[i_fft_bin][i_elevation][i_azimuth]
                beampattern[i_fft_bin][i_elevation][i_azimuth] = \
                    beamformers[i_fft_bin, :].transpose().conjugate().dot(v)
    return beampattern

def visualize_beampattern_1d(beampattern, scanning_azimuth_grid, frequency_bins, 
    signal_max_frequency, source_azimuths=None, title=None, figsize=(9, 6)):
    n_fft_bins = beampattern.shape[0]
    fig = plt.figure(figsize=(9, 6)); ax = fig.add_subplot(111)
    for i_bin in frequency_bins:
        ax.plot(scanning_azimuth_grid, beampattern[i_bin], 
                label="{:.0f} Hz".format(i_bin/n_fft_bins*signal_max_frequency));
    if source_azimuths is not None:
        for i_source in range(len(source_azimuths)):
            ax.axvline(x=source_azimuths[i_source], linestyle="--", label="Source angle");
    ax.set_xlim(scanning_azimuth_grid[0], scanning_azimuth_grid[-1]);
    ax.set_xlabel(r"Azimuth [degree]"); ax.set_ylabel("Beam pattern [dB]");
    ax.legend();
    if title is not None:
        ax.set_title(title)
    return fig, ax

def visualize_beampattern_1d_average(beampattern, scanning_azimuth_grid, frequency_range=None, 
    source_azimuths=None, title=None, figsize=(9, 6)):
    if frequency_range is None:
        ds_ave_beampattern = np.mean(np.abs(beampattern), axis=0)
    else:
        ds_ave_beampattern = np.mean(np.abs(beampattern[frequency_range[0]:frequency_range[1], :]), axis=0)

    ds_ave_beampattern_normalized = ds_ave_beampattern / np.max(ds_ave_beampattern)
    ds_ave_beampattern_normalized_db = 2 * to_db(ds_ave_beampattern_normalized)
    fig = plt.figure(figsize=(9, 6)); ax = fig.add_subplot(111)
    ax.plot(scanning_azimuth_grid, ds_ave_beampattern_normalized_db, label="Beam pattern");
    if source_azimuths is not None:
        for i_source in range(len(source_azimuths)):
            ax.axvline(x=source_azimuths[i_source], linestyle="--", label="Source angle");
    ax.set_xlim(scanning_azimuth_grid[0], scanning_azimuth_grid[-1]); ax.set_ylim(-60, 1)
    ax.set_xlabel(r"Azimuth [degree]"); ax.set_ylabel("Beam pattern [dB]");
    if title is not None:
        ax.set_title(title)
    ax.legend();
    return ax

def visualize_beampattern_2d(beampattern, scanning_azimuth_grid, signal_max_frequency, title=None, figsize=(9, 6)):
    fig = plt.figure(figsize=figsize); ax = fig.add_subplot(111)
    img = ax.imshow(beampattern, aspect="auto", origin="lower",
              extent=[scanning_azimuth_grid[0], scanning_azimuth_grid[-1], 0, signal_max_frequency*1e-3],
              cmap="coolwarm");
    ax.grid(False)
    ax.set_xlabel("Azimuth [degree]"); ax.set_ylabel("Frequency [kHz]");
    plt.colorbar(img);
    if title is not None:
        ax.set_title(title)
    return fig, ax

def compute_tf_beamformer_output(beamformer, tf_frames_multichannel, sampling_frequency, stft_params):
    n_samples_per_frame = stft_params["n_samples_per_frame"]
    n_fft_bins = stft_params["n_fft_bins"]
    hop_size = stft_params["hop_size"]
    stft_window = stft_params["window"]
    tf_out = np.zeros((n_fft_bins, tf_frames_multichannel.shape[2]), dtype=np.complex64)
    for i_fft_bin in range(n_fft_bins):
        tf_out[i_fft_bin] = beamformer[i_fft_bin].transpose().conjugate().dot(tf_frames_multichannel[i_fft_bin])
    t, out = istft(tf_out, fs=sampling_frequency, window=stft_window,
                             nperseg=n_samples_per_frame, noverlap=n_samples_per_frame-hop_size,
                             nfft=n_samples_per_frame, boundary=True)
    return out, tf_out, t

def compute_sinr_2(source_tf_multichannel, interference_tf_multichannel):
        source_power = 0
        interference_power = 0
        n_fft_bins = source_tf_multichannel.shape[0]
        for i_f in range(n_fft_bins):
            source_power += np.trace(source_stft_multichannel[i_f].dot(source_stft_multichannel[i_f].transpose().conjugate()))
            interference_power += np.trace(interference_stft_multichannel[i_f].dot(interference_stft_multichannel[i_f].transpose().conjugate()))
        psnr = source_power/interference_power
        psnr_db = to_db(psnr)
        return psnr_db, psnr
    
def compute_sinr(source_tf_multichannel, interference_tf_multichannel, weights=None):
    n_fft_bins, n_mics, _ = source_tf_multichannel.shape
    source_power = 0
    interference_power = 0
    if weights is not None:
        for i_f in range(n_fft_bins):
            source_power += weights[i_f].reshape(n_mics, 1).transpose().conjugate().dot(
                source_tf_multichannel[i_f].dot(
                source_tf_multichannel[i_f].transpose().conjugate())).dot(
                weights[i_f].reshape(n_mics, 1))
            interference_power += weights[i_f].transpose().conjugate().dot(
                interference_tf_multichannel[i_f].dot(
                interference_tf_multichannel[i_f].transpose().conjugate())).dot(
                weights[i_f])
    else:
        for i_f in range(n_fft_bins):
            source_power += np.trace(source_tf_multichannel[i_f].dot(source_tf_multichannel[i_f].transpose().conjugate()))
            interference_power += np.trace(interference_tf_multichannel[i_f].dot(interference_tf_multichannel[i_f].transpose().conjugate()))
    psnr = source_power/interference_power
    psnr_db = to_db(psnr)
    return psnr_db, psnr
    
def compute_mvdr_tf_beamformers(source_steering_vectors, tf_frames_multichannel, diagonal_loading_param=1):
    n_fft_bins, n_mics = source_steering_vectors.shape
    mvdr_tf_beamformers = np.zeros((n_fft_bins, n_mics), dtype=np.complex64)
    for i_fft_bin in range(n_fft_bins):
        n_frames = tf_frames_multichannel.shape[1]
        R = 1./n_frames * tf_frames_multichannel[i_fft_bin].dot(tf_frames_multichannel[i_fft_bin].transpose().conjugate()) \
                + diagonal_loading_param*np.identity(n_mics, dtype=np.complex64)
        invR = np.linalg.inv(R)
        normalization_factor = source_steering_vectors[i_fft_bin, :].transpose().conjugate().dot(invR).dot(source_steering_vectors[i_fft_bin, :])
        mvdr_tf_beamformers[i_fft_bin] = invR.dot(source_steering_vectors[i_fft_bin, :]) / (normalization_factor)
    return mvdr_tf_beamformers

def compute_mvndr_tf_beamformers(source_steering_vectors, tf_frames_multichannel, regularization_param=1):
    # Minimum variance near-distortless response beamformers
    # w = argmin w^H*R*w + \lambda * (v_s^H*w - 1)^2
    n_fft_bins, n_mics = source_steering_vectors.shape
    mvndr_tf_beamformers = np.zeros((n_fft_bins, n_mics), dtype=np.complex64)
    for i_fft_bin in range(n_fft_bins):
#         R = tf_frames_multichannel[i_fft_bin].dot(tf_frames_multichannel[i_fft_bin].transpose().conjugate()) + np.identity(n_mics)
#         invR = np.linalg.inv(R)
#         normalization_factor = source_steering_vectors[i_fft_bin, :].transpose().conjugate().dot(invR).dot(source_steering_vectors[i_fft_bin, :])
#         regularization_param = 1/normalization_factor
        R = tf_frames_multichannel[i_fft_bin].dot(tf_frames_multichannel[i_fft_bin].transpose().conjugate())\
            + np.identity(n_mics)\
            + regularization_param*source_steering_vectors[i_fft_bin, :]*source_steering_vectors[i_fft_bin, :].transpose().conjugate()
        invR = np.linalg.inv(R)
        mvndr_tf_beamformers[i_fft_bin] = regularization_param*invR.dot(source_steering_vectors[i_fft_bin, :])
    return mvndr_tf_beamformers

def compute_lcmv_tf_beamformers(steering_vectors, tf_frames_multichannel, constraint_vector):
    n_fft_bins, n_mics, n_steering_vectors = steering_vectors.shape
    lcmv_tf_beamformers = np.zeros((n_fft_bins, n_mics), dtype=np.complex64)
    for i_fft_bin in range(n_fft_bins):
        n_samples = len(tf_frames_multichannel[i_fft_bin])
        R = 1./n_samples * (tf_frames_multichannel[i_fft_bin].dot(
                    tf_frames_multichannel[i_fft_bin].transpose().conjugate()) \
                    + np.identity(n_mics) )
        invR = np.linalg.inv(R)
        normalization_matrix = steering_vectors[i_fft_bin].transpose().conjugate().dot(
            invR).dot(steering_vectors[i_fft_bin])
        normalization_matrix = (1 - 1e-3)*normalization_matrix \
                    + 1e-3*np.trace(normalization_matrix)/n_steering_vectors * 1*np.identity(n_steering_vectors)
        inverse_normalization_matrix = np.linalg.inv(normalization_matrix)
        lcmv_tf_beamformers[i_fft_bin] = invR.dot(steering_vectors[i_fft_bin]).dot(
            inverse_normalization_matrix).dot(constraint_vector)
    return lcmv_tf_beamformers

def compute_null_controlling_tf_beamformers(source_steering_vectors, null_steering_vectors, tf_frames_multichannel, 
        null_constraint_threshold, eigenvalue_percentage_threshold=0.99):
    n_fft_bins, n_mics, n_null_steering_vectors = null_steering_vectors.shape
    nc_tf_beamformers = np.zeros((n_fft_bins, n_mics), dtype=np.complex64)
    for i_fft_bin in range(n_fft_bins):
        
        null_steering_correlation_matrix = null_steering_vectors[i_fft_bin].dot(
            null_steering_vectors[i_fft_bin].transpose().conjugate())
        eigenvalues, eigenvectors = np.linalg.eigh(null_steering_correlation_matrix)
        running_sums = np.cumsum(np.abs(eigenvalues[-1::-1]))
        cutoff_index = np.searchsorted(running_sums, 
                                       eigenvalue_percentage_threshold * running_sums[-1])
        eigenvectors = eigenvectors[:, len(eigenvalues)-cutoff_index-1:]
        steering_vectors = np.hstack((source_steering_vectors[i_fft_bin].reshape(-1, 1), eigenvectors))
        n_samples = len(tf_frames_multichannel[i_fft_bin])
        R = 1./n_samples * (tf_frames_multichannel[i_fft_bin].dot(
                    tf_frames_multichannel[i_fft_bin].transpose().conjugate()) \
                    + np.identity(n_mics) )
        invR = np.linalg.inv(R)
        
        normalization_matrix = steering_vectors.transpose().conjugate().dot(
            invR).dot(steering_vectors)
        
        """ Regularization for dealing with ill-conditionaed normalization matrix
        Ref: Matthias Treder, Guido Nolte, "Source reconstruction of broadband EEG/MEG data using
the frequency-adaptive broadband (FAB) beamformer", bioRxiv
        Equation (12) in https://www.biorxiv.org/content/biorxiv/early/2018/12/20/502690.full.pdf
        """
        normalization_matrix = (1 - 1e-3)*normalization_matrix \
                    + 1e-3*np.trace(normalization_matrix)/steering_vectors.shape[1] * 10*np.identity(steering_vectors.shape[1])
        inverse_normalization_matrix = np.linalg.inv(normalization_matrix)
        
        constraint_vector = null_constraint_threshold*np.ones(steering_vectors.shape[1])
        constraint_vector[0] = 1
        
        nc_tf_beamformers[i_fft_bin] = invR.dot(steering_vectors).dot(
            inverse_normalization_matrix).dot(constraint_vector)
        
    return nc_tf_beamformers


def compute_null_controlling_tf_beamformers_2(source_steering_vectors, null_steering_vectors, tf_sample_covariance_batch, 
        null_constraint_threshold, eigenvalue_percentage_threshold=0.99, diagonal_loading_param=1):
    n_fft_bins, n_mics, n_null_steering_vectors = null_steering_vectors.shape
    nc_tf_beamformers = np.zeros((n_fft_bins, n_mics), dtype=np.complex64)
    for i_fft_bin in range(n_fft_bins):
        
        null_steering_correlation_matrix = null_steering_vectors[i_fft_bin].dot(
            null_steering_vectors[i_fft_bin].transpose().conjugate())
        eigenvalues, eigenvectors = np.linalg.eigh(null_steering_correlation_matrix)
        running_sums = np.cumsum(np.abs(eigenvalues[-1::-1]))
        cutoff_index = np.searchsorted(running_sums, 
                                       eigenvalue_percentage_threshold * running_sums[-1])
        eigenvectors = eigenvectors[:, len(eigenvalues)-cutoff_index-1:]
        steering_vectors = np.hstack((source_steering_vectors[i_fft_bin].reshape(-1, 1), eigenvectors))
        
        R = np.sum(tf_sample_covariance_batch[:, i_fft_bin, :, :], axis=0) / len(tf_sample_covariance_batch) + diagonal_loading_param*np.identity(n_mics)
        invR = np.linalg.inv(R)
        
        normalization_matrix = steering_vectors.transpose().conjugate().dot(
            invR).dot(steering_vectors)
        
        """ Regularization for dealing with ill-conditionaed normalization matrix
        Ref: Matthias Treder, Guido Nolte, "Source reconstruction of broadband EEG/MEG data using
the frequency-adaptive broadband (FAB) beamformer", bioRxiv
        Equation (12) in https://www.biorxiv.org/content/biorxiv/early/2018/12/20/502690.full.pdf
        """
        normalization_matrix = (1 - 1e-3)*normalization_matrix \
                    + 1e-3*np.trace(normalization_matrix)/steering_vectors.shape[1] * 1*np.identity(steering_vectors.shape[1])
        inverse_normalization_matrix = np.linalg.inv(normalization_matrix)
        
        constraint_vector = null_constraint_threshold*np.ones(steering_vectors.shape[1])
        constraint_vector[0] = 1
        
        nc_tf_beamformers[i_fft_bin] = invR.dot(steering_vectors).dot(
            inverse_normalization_matrix).dot(constraint_vector)
        
    return nc_tf_beamformers
def compute_null_controlling_minibatch_tf_beamformers(source_steering_vectors, 
        null_steering_vectors, tf_frames_multichannel_batch, 
        null_constraint_threshold, eigenvalue_percentage_threshold=0.99):
    n_fft_bins, n_mics, n_null_steering_vectors = null_steering_vectors.shape
    nc_tf_beamformers = np.zeros((n_fft_bins, n_mics), dtype=np.complex64)
    for i_fft_bin in range(n_fft_bins):
        null_steering_correlation_matrix = null_steering_vectors[i_fft_bin].dot(
            null_steering_vectors[i_fft_bin].transpose().conjugate())
        eigenvalues, eigenvectors = np.linalg.eigh(null_steering_correlation_matrix)
        running_sums = np.cumsum(np.abs(eigenvalues[-1::-1]))
        cutoff_index = np.searchsorted(running_sums, 
                                       eigenvalue_percentage_threshold * running_sums[-1])
        eigenvectors = eigenvectors[:, len(eigenvalues)-cutoff_index-1:]
        steering_vectors = np.hstack((source_steering_vectors[i_fft_bin].reshape(-1, 1), eigenvectors))
        R = np.zeros((n_mics, n_mics), dtype=np.complex64)
        for tf_frames_multichannel in tf_frames_multichannel_batch:            
            n_samples = len(tf_frames_multichannel[i_fft_bin])
            R += 1./n_samples * (tf_frames_multichannel[i_fft_bin].dot(
                        tf_frames_multichannel[i_fft_bin].transpose().conjugate()))
        R = R / len(tf_frames_multichannel_batch)
        R += 20*np.identity(n_mics) # To prevent singularity of R
        invR = np.linalg.inv(R)
        
        normalization_matrix = steering_vectors.transpose().conjugate().dot(
            invR).dot(steering_vectors)
        
        """ Regularization for dealing with ill-conditionaed normalization matrix
        Ref: Matthias Treder, Guido Nolte, "Source reconstruction of broadband EEG/MEG data using
the frequency-adaptive broadband (FAB) beamformer", bioRxiv
        Equation (12) in https://www.biorxiv.org/content/biorxiv/early/2018/12/20/502690.full.pdf
        """
        normalization_matrix = (1 - 1e-3)*normalization_matrix \
                    + 1e-3*np.trace(normalization_matrix)/steering_vectors.shape[1] * 10*np.identity(steering_vectors.shape[1])
        inverse_normalization_matrix = np.linalg.inv(normalization_matrix)
        
        constraint_vector = null_constraint_threshold*np.ones(steering_vectors.shape[1])
        constraint_vector[0] = 1
        
        nc_tf_beamformers[i_fft_bin] = invR.dot(steering_vectors).dot(
            inverse_normalization_matrix).dot(constraint_vector)
        
    return nc_tf_beamformers

def simulate_multichannel_tf(array_geometry, signal, theta, phi, sampling_frequency, stft_params):
    n_mics = len(array_geometry[0])
    n_samples_per_frame = stft_params["n_samples_per_frame"]
    n_fft_bins = stft_params["n_fft_bins"]
    hop_size = stft_params["hop_size"]
    stft_window = stft_params["window"]
    steering_vector = ( compute_steering_vectors(array_geometry, sampling_frequency, n_fft_bins, theta, phi) )[:, 0, 0, :]
    _, _, tf_frames = stft(signal.reshape(-1), fs=sampling_frequency, window=stft_window,
                             nperseg=n_samples_per_frame, noverlap=n_samples_per_frame-hop_size,
                             nfft=n_samples_per_frame, padded=True)
    tf_frames = tf_frames[:-1, 1:-1]
    tf_frames_multichannel = steering_vector.reshape(n_fft_bins, n_mics, 1)\
                                * tf_frames.reshape(tf_frames.shape[0], 1, tf_frames.shape[1])
    return tf_frames_multichannel

def simulate_multichannel_tf_mixtures(array_geometry, source,
        interferences, sampling_frequency, stft_params):
    source_signal = source["signal"]
    elevation_s = source["elevation"]
    azimuth_s = source["azimuth"]
    source_stft_multichannel = simulate_multichannel_tf(
        array_geometry, source_signal, elevation_s, azimuth_s,
        sampling_frequency, stft_params)
#     received_stft_multichannel = np.zeros(source_stft_multichannel.shape, dtype=np.complex64)

    received_stft_multichannel = source_stft_multichannel.copy()
    
    interference_stfts_multichannel_list = []
    for i_interference in range(len(interferences)):
        interference_signal = interferences[i_interference]["signal"]
        interference_elevation = interferences[i_interference]["elevation"]
        interference_azimuth = interferences[i_interference]["azimuth"]
        interference_stft_multichannel = simulate_multichannel_tf(array_geometry, interference_signal, 
                interference_elevation, interference_azimuth,
                sampling_frequency, stft_params)
        interference_stfts_multichannel_list.append(interference_stft_multichannel)        
    
    interference_stfts_multichannel = sum(interference_stfts_multichannel_list)
    received_stft_multichannel += interference_stfts_multichannel
    
    return received_stft_multichannel, source_stft_multichannel, interference_stfts_multichannel


def simulate_multichannel_tf_circular(array_geometry, signal, azimuth, sampling_frequency, stft_params):
    n_mics = len(array_geometry[0])
    n_samples_per_frame = stft_params["n_samples_per_frame"]
    n_fft_bins = stft_params["n_fft_bins"]
    hop_size = stft_params["hop_size"]
    stft_window = stft_params["window"]
    steering_vector = ( compute_steering_vectors_circular(array_geometry, sampling_frequency, stft_params, azimuth) )[:, 0, :]
    _, _, tf_frames = stft(signal.reshape(-1), fs=sampling_frequency, window=stft_window,
                             nperseg=n_samples_per_frame, noverlap=n_samples_per_frame-hop_size,
                             nfft=n_samples_per_frame, padded=True)
    tf_frames = tf_frames[:-1, 1:-1]
    tf_frames_multichannel = steering_vector.reshape(n_fft_bins, n_mics, 1)\
                                * tf_frames.reshape(tf_frames.shape[0], 1, tf_frames.shape[1])
    return tf_frames_multichannel

def check_distortless_constraint(weight, source_steering_vector, tolerance=1e-9):
    assert(np.abs(weight.transpose().conjugate().dot(source_steering_vector)) - 1 < tolerance)

def compute_steering_vectors_circular(array_geometry, sampling_frequency, stft_params, azimuth_grid):
    n_mics = len(array_geometry[0])
    n_azimuths = len(azimuth_grid)
    delay = np.zeros((n_azimuths, n_mics), dtype=np.float32)
    n_samples_per_frame = stft_params["n_samples_per_frame"]
    n_fft_bins = stft_params["n_fft_bins"]
    
    for m in range(n_mics):
        pos_x = array_geometry[0][m]
        pos_y = array_geometry[1][m]
        radius = np.sqrt(pos_x*pos_x + pos_y*pos_y)
        mic_azimuth = np.arctan2(pos_y, pos_x)
        for k in range(n_azimuths):
            azimuth = 2 * np.pi * azimuth_grid[k]/360
            delay[k][m] = - radius * np.cos(mic_azimuth - azimuth) * sampling_frequency / SOUND_SPEED
    steering_vectors = np.zeros((n_fft_bins, n_azimuths, n_mics), dtype=np.complex64)
    for i_fft_bin in range(n_fft_bins):
        v = 2 * np.pi * (i_fft_bin / n_samples_per_frame) * delay;
        steering_vectors[i_fft_bin] = np.cos(v) - np.sin(v) * 1j
    return steering_vectors

def compute_minimum_variance_tf_beamformers(source_steering_vectors, tf_frames_multichannel=None, diagonal_loading_param=1):
    n_fft_bins, n_mics = source_steering_vectors.shape
    mv_tf_beamformers = np.zeros((n_fft_bins, n_mics), dtype=np.complex64)
    for i_fft_bin in range(n_fft_bins):
        R = diagonal_loading_param*np.identity(n_mics, dtype=np.complex64)
        if tf_frames_multichannel is not None:
            n_frames = tf_frames_multichannel.shape[1]
            R += 1./n_frames * tf_frames_multichannel[i_fft_bin].dot(tf_frames_multichannel[i_fft_bin].transpose().conjugate())
        invR = np.linalg.inv(R)
        normalization_factor = source_steering_vectors[i_fft_bin].transpose().conjugate().dot(
            invR).dot(source_steering_vectors[i_fft_bin])
        mv_tf_beamformers[i_fft_bin] = invR.dot(source_steering_vectors[i_fft_bin]) / normalization_factor
    return mv_tf_beamformers

def compute_tf_beamformers(source_steering_vectors, beamformer_name="delaysum", 
    tf_frames_multichannel=None, diagonal_loading_param=1):
    n_fft_bins, n_mics = source_steering_vectors.shape
    if beamformer_name.lower() == "delaysum":
        tf_beamformer = compute_minimum_variance_tf_beamformers(
            source_steering_vectors, diagonal_loading_param=diagonal_loading_param)
    elif beamformer_name.lower() in ["mvdr", "mpdr"]:
        tf_beamformer = compute_minimum_variance_tf_beamformers(
            source_steering_vectors, tf_frames_multichannel, 
            diagonal_loading_param=diagonal_loading_param)
        
    return tf_beamformer

def prepare_multichannel_covariance_data(array_geometry, train_data, n_interferences, 
                                         training_angles, azimuth_step, 
                                         n_samples_each_config, sampling_frequency, 
                                         stft_params, filepath, random_seed=0, SAVE=False):
    if SAVE == True:
        import itertools
        training_elevations = training_angles["elevation"]
        training_azimuths = training_angles["azimuth"]
        n_interference_list = list(np.arange(n_interferences) + 1)
        training_interference_covariance = []
        n_mics = len(array_geometry[0])
        np.random.seed(0)
        for i_n_interference in tqdm(range(len(n_interference_list)), desc="Interference number"):
            n_interferences = n_interference_list[i_n_interference]
            interferences_params = []
            for i_interference in range(n_interferences):
                interference_params = list(itertools.product(*[training_elevations, training_azimuths]))
                interferences_params.append(interference_params)
            interferences_param_sets = list(itertools.product(*interferences_params))

            for i_param_set in tqdm(range(len(interferences_param_sets)), desc="Parameter set"):    
                param_set = interferences_param_sets[i_param_set]
                training_param_dict = {
                    "elevation": np.asarray(param_set).transpose()[0],
                    "azimuth": np.asarray(param_set).transpose()[1]
                    }

                # MORE TRAINING DATA IS BETTER            
                tf_sample_covariance_batch = np.zeros((n_samples_each_config, stft_params["n_fft_bins"], n_mics, n_mics), dtype=np.complex64)
                for i_training_sample in range(n_samples_each_config):
                    interference_signals = []
                    for i_interference in range(len(param_set)):
                        interference_signal = train_data[np.random.choice(len(train_data))]
                        interference_signals.append(interference_signal)                
                    interference_n_samples = min([len(signal) for signal in interference_signals])

                    interference_tf_multichannel_list = []
                    for i_interference in range(len(param_set)):
                        interference_signals[i_interference] = (interference_signals[i_interference])[0:interference_n_samples]
                        interference_elevation, interference_azimuth = param_set[i_interference]
                        interference_azimuth += 1*np.random.uniform()
                        interference_tf_multichannel = simulate_multichannel_tf(array_geometry, interference_signal, 
                                np.array([interference_elevation]), np.array([interference_azimuth]),
                                sampling_frequency, stft_params)
                        interference_tf_multichannel_list.append(interference_tf_multichannel)

                    received = sum(interference_tf_multichannel_list)
                    received_sample_covariance = np.zeros((stft_params["n_fft_bins"], n_mics, n_mics), dtype=np.complex64)
                    n_fft_bins, _, n_received_samples = received.shape
                    for i_fft_bin in range(n_fft_bins):
                        received_sample_covariance[i_fft_bin] = 1/n_received_samples * received[i_fft_bin].dot(received[i_fft_bin].transpose().conjugate())
                    tf_sample_covariance_batch[i_training_sample] = received_sample_covariance
                training_interference_covariance.append((tf_sample_covariance_batch, training_param_dict))
        with open(filepath, 'wb') as f:
            pickle.dump(training_interference_covariance, f, pickle.HIGHEST_PROTOCOL)
    else:
        with open(filepath, 'rb') as f:
            training_interference_covariance = pickle.load(f)
    return training_interference_covariance