Update: This repo has been modified to simulate optical distortion for thermal plumes in PIV, see
Bao, X & Lithgow-Bertelloni (2024). Self-correction of the optical distortion effect of thermal plumes in particle image velocimetry, Submitted

This code package implements the image generation methodology outlined in:

Rajendran, L. K., Bane, S., & Vlachos, P. (2019). PIV/BOS synthetic image generation in variable density environments for error analysis and experiment design. Measurement Science and Technology.

Please cite the above paper if you use this code package for your work.

UPDATED Instructions for running the software.
- Navigate to cuda_codes/Debug/
- make sure CUDA is installed, then run $which nvcc
	/usr/local/cuda-10.2/bin/nvcc
	Change all 10.1 to 10.2 in makefile_simplified, then
- Run "makefile_simplified" in the terminal - this should compile the codes and generate the file "libparallel_ray_tracing.so"
- Run "ls -lt libparallel_ray_tracing.so" and check the timestamp to ensure that the file has been recently updated.
- Type "export LD_LIBRARY_PATH=$(cd ../ && pwd)/lib:$(cd ../ && pwd)/lib64" in the terminal command
- Then type "echo $LD_LIBRARY_PATH" and confirm that it displays something like: "path-to-photon/cuda_codes/lib:path-to-photon/cuda_codes/lib64"
- To avoid performing the above two steps everytime, add the two commands to a bash initialization script (such as .bashrc or .profile)
- Navigate to python_codes
- Run "sample_run_script.sh piv" to run a sample piv simulation or "sample_run_script.sh bos" to run a sample bos simulation

Dependencies for Python:
- numpy
- scipy
- matplotlib
- libtiff
- ctypes
- pynrrd
Dependencies for CUDA (included in this repo):
- bz2 (https://www.sourceware.org/bzip2/)
- png (http://www.libpng.org/pub/png/libpng.html)
- teem (http://teem.sourceforge.net)
- cubic_interpolation_cuda (https://github.com/DannyRuijters/CubicInterpolationCUDA)

Notes on installation and code modification can be found in change_log. 
Sample data is provided in the sample-data/ directory (updated for Bao and Lithgow-Bertelloni 2004). Run the "sample_run_script.sh" file after compiling the software to run the package and generate sample images. You will have to make the bash script executable before running it. To do that, type: "chmod +x sample_run_script.sh" in the terminal. 

To reproduce the result in Bao and Lithgow-Bertelloni 2004, run "python batch_run_simulation.py ../sample-data/piv/parameters 1 1" under python_codes. You will see the result in sample-data/piv/images. 22.5cm_NNRD.py can be used to generate the refractive index field for frame 0 and 1 (plume_1.nrrd, plume_2.nrrd under sample-data/piv). "synthetic_2d.npz" under sample-data/ is the synthetic in-plane displacement, and two jupyter notebooks in the same directory are useful for changing ray tracing simulation paramters and how to use the synthetic flow data.

The script light_ray_processing.py contains useful python functions to load, manipulate, and display light ray data.
