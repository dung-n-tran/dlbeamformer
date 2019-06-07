# dlbeamformer
Robust Adaptive Beamforming using Dictionary Learning for Speech Enhancement

#### What's in here:
1. Benchmark and demo notebooks show how to use the beamformers. Use the benchmark script if you don't use Jupyter notebook.

2. File `dl_beamformers.py` implements Dictionary Learning based beamformers. Currently the benchmark and demo notebooks/scripts use `DictionaryLearningBeamformer` (for single data sample inputs) and "DLBeamformer` (for batch sample inputs).

3. File `dlbeamformers_utilities.py` implements tradictional beamformers such as Delay-Sum and MVDR along with helper functions for encapsulating immediate steps in computing these beamformers.

4. File `utilites.py` implements helper functions for stuff unrelated to beamformers.

5. File `config.INI` contains constants such as light speed and Short-Time-Fourier-Transform parameters. Need a better way to also contain other things such as array geometry.

6. Ignore the rest as they're on-going work.

#### How to use it:
* Follow with the benchmark and demo notebooks/scripts.


