import numpy as np
from dlbeamformer_utilities import compute_mvdr_tf_beamformers, check_distortless_constraint, compute_steering_vectors,\
compute_null_controlling_tf_beamformers, compute_null_controlling_minibatch_tf_beamformers,\
compute_null_controlling_tf_beamformers_2
from tqdm import tnrange, tqdm
from sklearn.linear_model import orthogonal_mp_gram
from omp import omp

class DictionaryLearningBeamformer(object):
    def __init__(self, array_geometry, sampling_frequency,
                 source_angles, stft_params, angle_grid, diagonal_loading_param=1, bf_type="NC"):
        """
        Parameters
        ----------
        array_geometry: 2-D numpy array describing the geometry of the microphone array
        sampling_frequency
        stft_params: Dictionary of STFT transform parameters including
            stft_params["n_samples_per_frame"]
            stft_params["n_fft_bins"]
            stft_params["hop_size"]
            stft_params["window"]
        bf_type: Type of the beamformer
        """
        self.array_geometry = array_geometry
        self.sampling_frequency = sampling_frequency
        self.source_angles = source_angles
        self.stft_params = stft_params
        self.angle_grid = angle_grid
        self.diagonal_loading_param = diagonal_loading_param
        self.bf_type = bf_type
        self.weights_ = None
        self.source_steering_vectors = self._compute_source_steering_vectors()
        self.steering_vectors = self._compute_steering_vectors()
        
    def _compute_source_steering_vectors(self):
        source_steering_vectors = []
        for i_source_angle, source_angle in enumerate(self.source_angles):
            v = compute_steering_vectors(self.array_geometry, 
                    self.sampling_frequency, self.stft_params["n_fft_bins"], 
                    source_angle["elevation"], source_angle["azimuth"])
            source_steering_vectors.append(v)
        return source_steering_vectors
    
    def _compute_steering_vectors(self):
        return compute_steering_vectors(self.array_geometry,
                    self.sampling_frequency, self.stft_params["n_fft_bins"],
                    self.angle_grid["elevation"], self.angle_grid["azimuth"])
    
    def _compute_weights(self, training_data, desired_null_width, 
            null_constraint_threshold, eigenvalue_percentage_threshold=0.99, batch_size=1, n_atoms_each_config=1):
        n_configurations = len(training_data)
        n_train_samples_each_config, n_fft_bins, n_mics, _ = training_data[0][0].shape
        n_sources = len(self.source_steering_vectors)
        D = np.zeros((n_sources, n_fft_bins, n_mics, n_configurations*n_atoms_each_config), dtype=complex)

        for i_source in range(n_sources):
            for i_configuration in tqdm(range(n_configurations), desc="Training configuration"):
                for i_atom in range(n_atoms_each_config):
                    batch_indices = np.random.choice(len(training_data[i_configuration][0]), batch_size, replace=True)
                    tf_sample_covariance_batch = training_data[i_configuration][0][batch_indices]
#                     print(training_data[i_configuration][0].shape, tf_sample_covariance_batch.shape, batch_indices)
                    null_azimuth_range = self._compute_null_angle_ranges(
                        training_data[i_configuration][1]["azimuth"], desired_null_width)
                    null_steering_vectors = compute_steering_vectors(
                        self.array_geometry, self.sampling_frequency,
                        self.stft_params["n_fft_bins"],
                        np.unique(training_data[i_configuration][1]["elevation"]), 
                        np.unique(null_azimuth_range)
                    )
                    null_steering_vectors = np.transpose(null_steering_vectors[:, :, 0, :], (0, 2, 1))
            
                    w = compute_null_controlling_tf_beamformers_2(
                            self.source_steering_vectors[i_source][:, 0, 0, :], null_steering_vectors, 
                            tf_sample_covariance_batch, 
                            null_constraint_threshold, 
                            eigenvalue_percentage_threshold=0.99, diagonal_loading_param=self.diagonal_loading_param)
                    D[i_source, :, :, i_configuration*n_atoms_each_config + i_atom] = w
            
        return D
    
    def _compute_null_angle_ranges(self, null_angles, desired_null_width):
        angle_ranges = []
        for null_angle in null_angles:
            angle_ranges.append(
                np.arange(null_angle - desired_null_width/2,
                          null_angle + desired_null_width/2, 0.1))
        return np.concatenate(angle_ranges)
            
#     def _initialize(self, X):
#         pass

    def _choose_weights(self, source_angle_index, x):
        weights_ = self.weights_[source_angle_index]
        n_fft_bins, n_mics, n_dictionary_atoms = weights_.shape
        min_ave_energy = np.inf
        optimal_weight_index = None
        for i_dictionary_atom in range(n_dictionary_atoms):
            w_frequency = weights_[:, :, i_dictionary_atom]
            energy = 0
            n_fft_bins = w_frequency.shape[0]
            for i_fft_bin in range(n_fft_bins):
                w = w_frequency[i_fft_bin]
                R = x[i_fft_bin].dot(x[i_fft_bin].transpose().conjugate())
                energy += np.real(w.transpose().conjugate().dot(R).dot(w))
            ave_energy = energy / n_fft_bins
            if min_ave_energy > ave_energy:
                min_ave_energy = ave_energy
                optimal_weight_index = i_dictionary_atom
        optimal_weights = weights_[:, :, optimal_weight_index]
#         optimal_weights = np.zeros((n_fft_bins, n_mics), dtype=np.complex64)
# #         for i_fft_bin in tqdm(range(n_fft_bins), desc="FFT bin"):
#         for i_fft_bin in range(n_fft_bins):
#             R = x[i_fft_bin].dot(x[i_fft_bin].transpose().conjugate()) + 1*self.diagonal_loading_param*np.identity(n_mics)
#             W = weights_[i_fft_bin]
#             i_fft_optimal_weight_index = np.argmin(np.diagonal(np.abs(W.transpose().conjugate().dot(
#                 R).dot(W))))
#             optimal_weights[i_fft_bin] = weights_[i_fft_bin, :, i_fft_optimal_weight_index]
        return optimal_weights
    
    def fit(self, training_data, desired_null_width, 
            null_constraint_threshold, eigenvalue_percentage_threshold=0.99, batch_size=1, n_atoms_each_config=1):
        """
        Parameters
        ----------
        """
        D = self._compute_weights(training_data, desired_null_width, 
            null_constraint_threshold, eigenvalue_percentage_threshold, batch_size, n_atoms_each_config)
        self.weights_ = D
        return self

    def choose_weights(self, source_angle_index, x):
        return self._choose_weights(source_angle_index, x)
    

class DLBeamformer(object):
    def __init__(self, array_geometry, sampling_frequency,
                 source_angles, stft_params, angle_grid, diagonal_loading_param=1,
                 n_dict_atoms=None, n_nonzero_coefficients=None, 
                 n_train_max_iterations=100, train_error_tolerance=1e-6, bf_type=None):
        """
        Parameters
        ----------
        array_geometry: 2-D numpy array describing the geometry of the microphone array
        sampling_frequency
        stft_params: Dictionary of STFT transform parameters including
            stft_params["n_samples_per_frame"]
            stft_params["n_fft_bins"]
            stft_params["hop_size"]
            stft_params["window"]
        bf_type: Type of the beamformer
        """
        print("Initialize DLBeamformer")
        self.array_geometry = array_geometry
        self.sampling_frequency = sampling_frequency
        self.source_angles = source_angles
        self.stft_params = stft_params
        self.angle_grid = angle_grid
        self.diagonal_loading_param = diagonal_loading_param
        self.bf_type = bf_type
        self.weights_ = None
        self.source_steering_vectors = self._compute_source_steering_vectors()
        self.steering_vectors = self._compute_steering_vectors()
        self.n_dict_atoms = n_dict_atoms
        self.n_train_max_iterations = n_train_max_iterations
        self.n_nonzero_coefficients = n_nonzero_coefficients
        self.train_error_tolerance = train_error_tolerance
        self.training_loss = []
        
    def _compute_source_steering_vectors(self):
        source_steering_vectors = []
        for i_source_angle, source_angle in enumerate(self.source_angles):
            v = compute_steering_vectors(self.array_geometry, 
                    self.sampling_frequency, self.stft_params["n_fft_bins"], 
                    source_angle["elevation"], source_angle["azimuth"])
            source_steering_vectors.append(v)
        return source_steering_vectors
    
    def _compute_steering_vectors(self):
        return compute_steering_vectors(self.array_geometry,
                    self.sampling_frequency, self.stft_params["n_fft_bins"],
                    self.angle_grid["elevation"], self.angle_grid["azimuth"])
    
    def _compute_weights(self, training_data, desired_null_width, 
            null_constraint_threshold, eigenvalue_percentage_threshold=0.99, 
            batch_size=1, n_train_batches_each_config=1):
        n_configurations = len(training_data)
        n_train_samples_each_config, n_fft_bins, n_mics, _ = training_data[0][0].shape
        n_sources = len(self.source_steering_vectors)
        
        if self.n_dict_atoms == None:
                self.n_dict_atoms = n_configurations
        # Initialization        
        dictionary = np.random.randn(n_sources, n_fft_bins, n_mics, self.n_dict_atoms) + \
                    1j*np.random.randn(n_sources, n_fft_bins, n_mics, self.n_dict_atoms)
        coefficients = np.zeros((n_sources, n_fft_bins, self.n_dict_atoms, 
                                 n_configurations*n_train_batches_each_config), 
                                 dtype=np.complex64)
        
        # Compute desired weights
        desired_weights = np.zeros((n_sources, n_fft_bins, n_mics,
                                    n_configurations*n_train_batches_each_config),
                                    dtype=np.complex64)
        # for i_configuration in tqdm(range(n_configurations), desc="     Training configuration"):
        for i_source in range(n_sources):
            for i_configuration in range(n_configurations):
                for i_batch in range(n_train_batches_each_config):
                    # Get a batch of data
                    batch_indices = np.random.choice(len(training_data[i_configuration][0]), batch_size, replace=True)
                    tf_sample_covariance_batch = training_data[i_configuration][0][batch_indices]

                    # Compute null steering vectors for nulling constraints
                    null_azimuth_range = self._compute_null_angle_ranges(
                            training_data[i_configuration][1]["azimuth"], desired_null_width)
                    null_steering_vectors = compute_steering_vectors(
                        self.array_geometry, self.sampling_frequency,
                        self.stft_params["n_fft_bins"],
                        np.unique(training_data[i_configuration][1]["elevation"]), 
                        np.unique(null_azimuth_range)
                    )
                    null_steering_vectors = np.transpose(null_steering_vectors[:, :, 0, :], (0, 2, 1))

                    # Compute desired weights for the selected batch of data
                    w = compute_null_controlling_tf_beamformers_2(
                            self.source_steering_vectors[i_source][:, 0, 0, :], null_steering_vectors, 
                            tf_sample_covariance_batch, 
                            null_constraint_threshold, 
                            eigenvalue_percentage_threshold=0.99, diagonal_loading_param=self.diagonal_loading_param)
                    desired_weights[i_source, :, :, i_configuration*n_train_batches_each_config + i_batch] = w
                
        # Training loop
        for i_source in range(n_sources):
            for i_train_iteration in tqdm(range(self.n_train_max_iterations), desc="Training iteration"):
                # Each config
                i_iteration_train_loss = 0
                
                # Update sparse coefficients given the dictionary
                for i_fft_bin in range(n_fft_bins):
                    for i_sample in range(desired_weights.shape[3]):
                        coefficients[i_source, i_fft_bin, :, i_sample] = omp(
                                    dictionary[i_source][i_fft_bin], 
                                    desired_weights[i_source, i_fft_bin, :, i_sample], 
                                    nonneg=False, ncoef=self.n_nonzero_coefficients, 
                                    tol=self.train_error_tolerance, verbose=False
                                ).coef
                # Update dictionary given the sparse coeficients                    
                for i_fft_bin in range(n_fft_bins):    
                    ### Update dictionary
                    dictionary[i_source][i_fft_bin] = desired_weights[i_source][i_fft_bin].dot(
                        np.linalg.pinv(coefficients[i_source][i_fft_bin])
                    )
                for i_fft_bin in range(n_fft_bins):
                    i_iteration_train_loss += 0.5*np.linalg.norm(dictionary[i_source][i_fft_bin].dot(coefficients[i_source][i_fft_bin]) - \
                        desired_weights[i_source][i_fft_bin])**2
                i_iteration_train_loss = i_iteration_train_loss / n_fft_bins
                print("\t\tTrain loss at current iteration {:.9f}".format(i_iteration_train_loss))
                self.training_loss.append(i_iteration_train_loss)
                    
        return dictionary, coefficients, desired_weights
    
    def _compute_null_angle_ranges(self, null_angles, desired_null_width):
        angle_ranges = []
        for null_angle in null_angles:
            angle_ranges.append(
                np.arange(null_angle - desired_null_width/2,
                          null_angle + desired_null_width/2, 0.1))
        return np.concatenate(angle_ranges)
            
#     def _initialize(self, X):
#         pass

    def _choose_weights(self, source_angle_index, x):
        weights_ = self.weights_[source_angle_index]
        n_fft_bins, n_mics, n_dictionary_atoms = weights_.shape
        min_ave_energy = np.inf
        optimal_weight_index = None
        for i_dictionary_atom in range(n_dictionary_atoms):
            w_frequency = weights_[:, :, i_dictionary_atom]
            energy = 0
            n_fft_bins = w_frequency.shape[0]
            for i_fft_bin in range(n_fft_bins):
                w = w_frequency[i_fft_bin]
                R = x[i_fft_bin].dot(x[i_fft_bin].transpose().conjugate())
                energy += np.real(w.transpose().conjugate().dot(R).dot(w))
            ave_energy = energy / n_fft_bins
            if min_ave_energy > ave_energy:
                min_ave_energy = ave_energy
                optimal_weight_index = i_dictionary_atom
        optimal_weights = weights_[:, :, optimal_weight_index]
#         optimal_weights = np.zeros((n_fft_bins, n_mics), dtype=np.complex64)
# #         for i_fft_bin in tqdm(range(n_fft_bins), desc="FFT bin"):
#         for i_fft_bin in range(n_fft_bins):
#             R = x[i_fft_bin].dot(x[i_fft_bin].transpose().conjugate()) + 1*self.diagonal_loading_param*np.identity(n_mics)
#             W = weights_[i_fft_bin]
#             i_fft_optimal_weight_index = np.argmin(np.diagonal(np.abs(W.transpose().conjugate().dot(
#                 R).dot(W))))
#             optimal_weights[i_fft_bin] = weights_[i_fft_bin, :, i_fft_optimal_weight_index]
        return optimal_weights
    
    def fit(self, training_data, desired_null_width, 
            null_constraint_threshold, eigenvalue_percentage_threshold=0.99, 
            batch_size=1, n_train_batches_each_config=1):
        """
        Parameters
        ----------
        """
        D = self._compute_weights(training_data, desired_null_width, 
            null_constraint_threshold, eigenvalue_percentage_threshold, 
            batch_size, n_train_batches_each_config)
        self.weights_ = D
        return self

    def choose_weights(self, source_angle_index, x):
        return self._choose_weights(source_angle_index, x)
    
    
class DLBatchBeamformer(object):
    def __init__(self, array_geometry, sampling_frequency,
                 source_angles, stft_params, angle_grid, bf_type="NC"):
        """
        Parameters
        ----------
        array_geometry: 2-D numpy array describing the geometry of the microphone array
        sampling_frequency
        stft_params: Dictionary of STFT transform parameters including
            stft_params["n_samples_per_frame"]
            stft_params["n_fft_bins"]
            stft_params["hop_size"]
            stft_params["window"]
        bf_type: Type of the beamformer
        """
        print("Initialize DL Batch Beamformer")
        self.array_geometry = array_geometry
        self.sampling_frequency = sampling_frequency
        self.source_angles = source_angles
        self.stft_params = stft_params
        self.angle_grid = angle_grid
        self.bf_type = bf_type
        self.weights_ = None
        self.source_steering_vectors = self._compute_source_steering_vectors()
        self.steering_vectors = self._compute_steering_vectors()
        
    def _compute_source_steering_vectors(self):
        source_steering_vectors = []
        for i_source_angle, source_angle in enumerate(self.source_angles):
            v = compute_steering_vectors(self.array_geometry, 
                    self.sampling_frequency, self.stft_params["n_fft_bins"], 
                    source_angle["theta"], source_angle["phi"])
            source_steering_vectors.append(v)
        return source_steering_vectors
    
    def _compute_steering_vectors(self):
        return compute_steering_vectors(self.array_geometry,
                    self.sampling_frequency, self.stft_params["n_fft_bins"],
                    self.angle_grid["theta"], self.angle_grid["phi"])
    
    def _compute_weights(self, training_data, desired_null_width, 
            null_constraint_threshold, eigenvalue_percentage_threshold=0.99, 
            batch_size=1, n_atoms_each_config=1):
        n_configurations = len(training_data)
        n_fft_bins, n_mics, _ = training_data[0][0][0].shape
        n_sources = len(self.source_steering_vectors)
        D = np.zeros((n_sources, n_fft_bins, n_mics, n_configurations*n_atoms_each_config), dtype=complex)
        for i_source in range(n_sources):
            for i_configuration in tqdm(range(n_configurations), desc="Configuration"):
                for i_atom in range(n_atoms_each_config):
                    train_data_indices = np.random.choice(len(training_data[i_configuration][0]), batch_size)
                    tf_sample_covariance_matrices = training_data[i_configuration][0][train_data_indices]
                    null_angle_range = self._compute_null_angle_ranges(
                        training_data[i_configuration][1]["theta"], desired_null_width)
                    null_steering_vectors = compute_steering_vectors(
                        self.array_geometry, self.sampling_frequency,
                        self.stft_params["n_fft_bins"],
                        np.unique(null_angle_range), np.unique(training_data[i_configuration][1]["phi"])
                    )
                    null_steering_vectors = np.transpose(null_steering_vectors[:, :, 0, :], (0, 2, 1))
                    w = compute_null_controlling_minibatch_tf_beamformers(
                            self.source_steering_vectors[i_source][:, 0, 0, :], null_steering_vectors, 
                            tf_sample_covariance_matrices, 
                            null_constraint_threshold, 
                            eigenvalue_percentage_threshold=0.99)
                    D[i_source, :, :, i_configuration*n_atoms_each_config + i_atom] = w
            
        return D
    
    def _compute_null_angle_ranges(self, null_thetas, desired_null_width):
        theta_ranges = []
        for null_theta in null_thetas:
            theta_ranges.append(
                np.arange(null_theta - desired_null_width/2,
                          null_theta + desired_null_width/2, 0.1))
        return np.concatenate(theta_ranges)
            
#     def _initialize(self, X):
#         pass

    def _choose_weights(self, source_angle_index, x):
        weights_ = self.weights_[source_angle_index]
        n_fft_bins, n_mics, n_dictionary_atoms = weights_.shape
#         min_ave_energy = np.inf
#         optimal_weight_index = None
#         for i_dictionary_atom in range(n_dictionary_atoms):
#             w_frequency = weights_[:, :, i_dictionary_atom]
#             energy = 0
#             n_fft_bins = w_frequency.shape[0]
#             for i_fft_bin in range(n_fft_bins):
#                 w = w_frequency[i_fft_bin]
#                 R = x[i_fft_bin].dot(x[i_fft_bin].transpose().conjugate())
#                 energy += np.real(w.transpose().conjugate().dot(R).dot(w))
#             ave_energy = energy / n_fft_bins
#             if min_ave_energy > ave_energy:
#                 min_ave_energy = ave_energy
#                 optimal_weight_index = i_dictionary_atom
#         optimal_weight = weights_[:, :, optimal_weight_index]
        optimal_weights = np.zeros((n_fft_bins, n_mics), dtype=np.complex64)
        for i_fft_bin in tqdm(range(n_fft_bins), desc="FFT bin"):
            R = x[i_fft_bin].dot(x[i_fft_bin].transpose().conjugate())
            W = weights_[i_fft_bin]
            i_fft_optimal_weight_index = np.argmin(np.diagonal(np.abs(W.transpose().conjugate().dot(
                R).dot(W))))
            optimal_weights[i_fft_bin] = weights_[i_fft_bin, :, i_fft_optimal_weight_index]
        return optimal_weights
    
    def fit(self, training_data, desired_null_width, 
            null_constraint_threshold, eigenvalue_percentage_threshold=0.99, 
            batch_size=1, n_atoms_each_config=1):
        """
        Parameters
        ----------
        """
        D = self._compute_weights(training_data, desired_null_width, 
            null_constraint_threshold, eigenvalue_percentage_threshold, 
            batch_size, n_atoms_each_config)
        self.weights_ = D
        return self

    def choose_weights(self, source_angle_index, x):
        return self._choose_weights(source_angle_index, x)