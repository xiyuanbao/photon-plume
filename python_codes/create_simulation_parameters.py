import numpy as np
import scipy.io as sio

def create_simulation_parameters(simulation_type):
    # % This function is designed to create a basic parameters structure for
    # % controlling simulating PIV data using the thick lens camera simulation
    # % code.

    # % This initializes the simulation parameters structure
    simulation_parameters = {}

    simulation_parameters['simulation_type'] = simulation_type
    # %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    # % Lens Design Parameters                                                  %
    # %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

    # % This initializes the lens design parameters structure
    simulation_parameters['lens_design'] = {}
    # % This adds the lens focal length to the design structure (in microns)
    simulation_parameters['lens_design']['focal_length'] = 105e3
    # % This adds the lens f/# to the design structure
    simulation_parameters['lens_design']['aperture_f_number'] = 8.0
    # % This adds the object distance of the lens (ie the distance between the
    # % lens front principal plane and the center of the focal plane) to the
    # % design structure (in microns)
    simulation_parameters['lens_design']['object_distance'] = 700e3
    simulation_parameters['lens_design']['lens_radius_of_curvature'] = 100e3 #100000.0e3
    simulation_parameters['lens_design']['lens_model'] = 'general'
    # fraction of the lens frontal radius that will be used for imaging
    simulation_parameters['lens_design']['ray_cone_pitch_ratio'] = 1e-4
    # %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    # % Camera Design Parameters                                                %
    # %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    
    # % This adds the camera design parameters structure
    simulation_parameters['camera_design'] = {}
    # % This is the pixel pitch (in microns)
    simulation_parameters['camera_design']['pixel_pitch'] = 17
    # % This is the number of pixels in the x-direction
    simulation_parameters['camera_design']['x_pixel_number'] = 1024
    # % This is the number of pixels in the y-direction
    simulation_parameters['camera_design']['y_pixel_number'] = 1024
    # % This is the bit depth of the camera sensor (which must be an integer
    # % less then or equal to 16)
    simulation_parameters['camera_design']['pixel_bit_depth'] = 10
    # % This is the gain of the sensor in decibels
    simulation_parameters['camera_design']['pixel_gain'] = []
    # % This is the x angle of the camera to the particle volume
    simulation_parameters['camera_design']['x_camera_angle'] = -0.00 * np.pi / 180.0
    # % This is the y angle of the camera to the particle volume
    simulation_parameters['camera_design']['y_camera_angle'] = -0.00 * np.pi / 180.0
    # this is the image noise to add
    simulation_parameters['camera_design']['image_noise'] = 0.00
    # option to turn on scaling intensity to full bit depth
    simulation_parameters['camera_design']['intensity_rescaling'] = True
    if simulation_type == 'piv':
        # %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        # % Particle Field Simulation Parameters                                    %
        # %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        #
        # % This adds the particle field parameters structure
        simulation_parameters['particle_field'] = {}
        # % This adds the Boolean value stating whether to generate the particle
        # % field images to the structure
        simulation_parameters['particle_field']['generate_particle_field_images'] = True
        # % This adds the directory containing the particle locations to the
        # % parameters structure
        simulation_parameters['particle_field']['load_particle_data'] = False
        # %simulation_parameters['particle_field']data_directory='/mnt/current_storage/Projects2/Tomo_PIV/Camera_Simulation_GUI/Test_Particle_Data/';
        simulation_parameters['particle_field']['data_directory'] = ''
        # % This adds the prefix of the particle data filenames to the parameters
        # % structure
        simulation_parameters['particle_field']['data_filename_prefix'] = 'particle_data_frame'
        # % This adds the vector giving the frames of particle positions to load to
        # % the parameters structure (this indexes into the list generated by the
        # % command 'dir([data_directory,data_filename_prefix,'*.mat'])')
        # %simulation_parameters['particle_field']frame_vector=4:4;
        simulation_parameters['particle_field']['frame_vector'] = range(1, 3)  # range goes up to stop-1
        # % This is the number of particles to simulate out of the list of possible
        # % particles (if this number is larger than the number of saved particles,
        # % an error will be returned)
        simulation_parameters['particle_field']['particle_number'] = 5e5
        # % This is the number of lightrays to simulate per particle (this is roughly
        # % equivalent to the power of the laser)
        simulation_parameters['particle_field'][
            'lightray_number_per_particle'] = 1e4  # % 1e4 is a good number of lightrays . . .
        # % This is the number of lightrays to propogate per iteration (this is a
        # % function of the RAM available on the computer)
        simulation_parameters['particle_field']['lightray_process_number'] = 1e6
        # % This is the gain of the sensor in decibels to be used in the particle
        # % field simulation
        simulation_parameters['particle_field']['pixel_gain'] = 30
        # % This is the Full Width Half Maximum of the laser sheet Gaussian function
        # % (in microns) which will produce an illuminated sheet on the XY plane
        simulation_parameters['particle_field']['gaussian_beam_fwhm'] = 0.73e3
        # % This is a Boolean value stating whether to perform Mie scattering
        # % simulation
        simulation_parameters['particle_field']['perform_mie_scattering'] = True
        # % This is the refractive index of the medium in which the particles are
        # % seeded (typically either water or air)
        simulation_parameters['particle_field']['medium_refractive_index'] = 1.3330
        # % This is the refractive index of the seeding particles used in the
        # % simulation
        simulation_parameters['particle_field']['particle_refractive_index'] = 1.5700
        # % This is the mean diameter of the particles being used in the simulated
        # % experiment (in microns) - the arithmetic mean of the particle diameters
        # % will typically be slightly smaller than this value due to the use of a
        # % log-normal distribution in the particle sizes
        simulation_parameters['particle_field']['particle_diameter_mean'] = 27
        # % This is the standard deviation of the particle diameter used in the
        # % simulated experiment (in microns) - the arithmetic standard deviation
        # % will typically be slightly smaller than this value (this effect gets
        # % larger the closer the standard deviation gets to the mean) due to the use
        # % of a log-normal distribution in particle sizes
        simulation_parameters['particle_field']['particle_diameter_std'] = 5
        # % This is the number of different particle sizes to model since the
        # % particle diameters are taken in discrete intervals for computational
        # % efficiency
        simulation_parameters['particle_field']['particle_diameter_number'] = 27
        # % This is the cutoff threshhold of the log-normal cumulative density
        # % function beyond which extrema particle diameters are not calculated (ie
        # % if this is set to 0.01 then 1% of the possible particle diameters both
        # % much smaller  and much larger than the mean diameter that would be found
        # % on a continuous particle diameter range will not be included in the
        # % simulation)
        simulation_parameters['particle_field']['particle_diameter_cdf_threshhold'] = 0.01
        # % This is the number of angles to calculate the Mie scattering intensity
        # % over (which is later interpolated to the precise angles for each paricle)
        simulation_parameters['particle_field']['mie_scattering_angle_number'] = 128
        # % This is a direction vector (ie the magnitude doesn't matter) that points
        # % in the direction of the laser beam propogation - this vector (at least
        # % for now), is defined by a 1 x 3 array and lies in the XY plane (ie the # % last component must be zero)
        simulation_parameters['particle_field']['beam_propogation_vector'] = np.array([0.0, 1.0, 0.0])
        # % This is the wavelength of the laser used for illumination of the
        # % particles (in microns)
        simulation_parameters['particle_field']['beam_wavelength'] = 0.532

    elif simulation_type == 'calibration':
        # %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        # % Calibration Grid Parameters                                             %
        # %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

        # % This creates the calibration grid parameters structure
        simulation_parameters['calibration_grid'] = {}
        # % This adds the Boolean value stating whether to generate the calibration
        # % images to the structure
        simulation_parameters['calibration_grid']['generate_calibration_grid_images'] = True
        # % This adds the grid point diameter to the structure
        simulation_parameters['calibration_grid']['grid_point_diameter'] = 3.2e3
        # % This adds the grid point spacing to the structure
        simulation_parameters['calibration_grid']['x_grid_point_spacing'] = 15e3
        simulation_parameters['calibration_grid']['y_grid_point_spacing'] = 15e3
        # % This adds the grid point number to the calibration structure
        simulation_parameters['calibration_grid']['x_grid_point_number'] = 11
        simulation_parameters['calibration_grid']['y_grid_point_number'] = 11
        # % This adds the calibration plane number to the structure
        simulation_parameters['calibration_grid']['calibration_plane_number'] = 7
        # % This adds the calibration plane spacing to the structure
        simulation_parameters['calibration_grid']['calibration_plane_spacing'] = 1e3
        # % This adds the number of 'particles' (ie lightray source points) per grid
        # % point to the calibration structure
        simulation_parameters['calibration_grid']['particle_number_per_grid_point'] = 1e3
        # % This is the number of lightrays to simulate per 'particle' (ie lightray
        # % source point) in the calibration grid
        simulation_parameters['calibration_grid']['lightray_number_per_particle'] = 5e2
        # % This is the number of lightrays to propogate per iteration (this is a
        # % function of the RAM available on the computer)
        simulation_parameters['calibration_grid']['lightray_process_number'] = 1e6
        # % This is the gain of the sensor in decibels to be used in the calibration
        # % grid simulation
        simulation_parameters['calibration_grid']['pixel_gain'] = 25

    elif simulation_type == 'bos':
        # %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        # % BOS pattern Parameters                                             %
        # %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

        # % This creates the calibration grid parameters structure
        simulation_parameters['bos_pattern'] = {}
        # % This adds the Boolean value stating whether to generate the calibration
        # % images to the structure
        simulation_parameters['bos_pattern']['generate_bos_pattern_images'] = True
        # % This adds the grid point diameter to the structure
        simulation_parameters['bos_pattern']['grid_point_diameter'] = 6.0e2
        # % This adds the grid point number to the calibration structure
        simulation_parameters['bos_pattern']['x_grid_point_number'] = 150
        simulation_parameters['bos_pattern']['y_grid_point_number'] = 150
        simulation_parameters['bos_pattern']['grid_point_number'] = 1000
        # % This adds the number of 'particles' (ie lightray source points) per grid
        # % point to the calibration structure
        simulation_parameters['bos_pattern']['particle_number_per_grid_point'] = 100
        # % This is the number of lightrays to simulate per 'particle' (ie lightray
        # % source point) in the calibration grid
        simulation_parameters['bos_pattern']['lightray_number_per_particle'] = 5e2
        # % This is the number of lightrays to propogate per iteration (this is a
        # % function of the RAM available on the computer)
        simulation_parameters['bos_pattern']['lightray_process_number'] = 1e6
        # % This is the gain of the sensor in decibels to be used in the calibration
        # % grid simulation
        simulation_parameters['bos_pattern']['pixel_gain'] = 25
        # This sets the minimum and maximum values of the X co-ordinate of the bos pattern target
        simulation_parameters['bos_pattern']['X_Min'] = -7.5e4
        simulation_parameters['bos_pattern']['X_Max'] = +7.5e4
        # This sets the minimum and maximum values of the Y co-ordinate of the bos pattern target
        simulation_parameters['bos_pattern']['Y_Min'] = -7.5e4
        simulation_parameters['bos_pattern']['Y_Max'] = +7.5e4
        # number of light ray positions and directions to save
        simulation_parameters['bos_pattern']['num_lightrays_save'] = 1e6

    # %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    # % Density Gradient Parameters
    # %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    simulation_parameters['density_gradients'] = {}

    simulation_parameters['density_gradients']['simulate_density_gradients'] = False
    # This specifies the path to the file containing the density gradient data
    simulation_parameters['density_gradients']['density_gradient_filename'] = ''
    # ray tracing algorithm (1 - euler, 2 - rk4, 3 - rk45, 4 - adams-bashforth)
    simulation_parameters['density_gradients']['ray_tracing_algorithm'] = 1

    # %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    # % Image Writing Parameters                                                %
    # %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

    # % This adds the output data parameters structure
    simulation_parameters['output_data'] = {}
    # % This adds the directory to save the images to parameters
    # % structure
    # %simulation_parameters.output_data.particle_image_directory='/mnt/current_storage/Projects2/Tomo_PIV/Camera_Simulation_GUI/camera_simulation_package_01/test_directory/particle_images/';
    simulation_parameters['output_data']['image_directory'] = ''

    # option to crop final image
    simulation_parameters['output_data']['crop_image'] = False
    # option to save light ray
    simulation_parameters['output_data']['save_lightrays'] = False
    # number of light rays to be saved
    simulation_parameters['output_data']['num_lightrays_save'] = 100
    # option to save light ray positions inside density gradient volume (for each light ray that is being saved)
    simulation_parameters['output_data']['save_intermediate_ray_data'] = False
    # number of intermediate light ray positions to be saved (for each light ray)
    simulation_parameters['output_data']['num_intermediate_positions_save'] = 100
    return simulation_parameters

