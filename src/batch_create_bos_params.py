import create_bos_simulation_parameters as CBS
import numpy as np
import scipy.io as sio

# set filepath
filepath = '/home/barracuda/a/lrajendr/Projects/camera_simulation/data/bos_parameters/dot-size/'

# set base file name
filename_base = 'bos_simulation_parameters_'

# set expected displacement
expected_displacement_pixels = 2 

# create array of dot sizes in pixels
# dot_size_pixels = np.array([0.1, 0.5, 1.0]) * expected_displacement_pixels
dot_size_pixels = np.linspace(.05, .5, num=10, endpoint=True) * 10
#dot_size_pixels = np.array([0.2]) * 10

print dot_size_pixels
# set scaling factor to convert from pixels to microns (from trial and error)
pixels_to_microns = 1e2

# calculate dot size in microns
dot_size_microns = dot_size_pixels * pixels_to_microns

for i in range(0,len(dot_size_microns)):
    # call function to create params for each dot size
    piv_simulation_parameters = CBS.create_bos_simulation_parameters()

    # modify the dot size parameter
    piv_simulation_parameters['bos_pattern']['grid_point_diameter'] = dot_size_microns[i]

    # modify the number of dots along x
    nx = 150
    piv_simulation_parameters['bos_pattern']['x_grid_point_number'] = nx
    
    # modify the number of dots along y
    ny = 150
    piv_simulation_parameters['bos_pattern']['y_grid_point_number'] = ny

    # modify the f# of the lens
    f_number = 16
    piv_simulation_parameters['lens_design']['aperture_f_number'] = f_number
    
    ## modify the object distance
    #piv_simulation_parameters['lens_design']['object_distance'] = 680e3

    # modify the file containing the density gradient data
    piv_simulation_parameters['density_gradients']['density_gradient_filename'] = '/home/barracuda/a/lrajendr/Projects/parallel_ray_tracing/data/const_grad_BOS_grad_x_02.nrrd'
    
    '''
    # modify the number of particle that will be used for each grid point (more particles for larger grid points)
    # the multiplying constant is the number of particles for each micron and has been determined from experience. this
    # could vary from one optical setup to the other
    piv_simulation_parameters['bos_pattern']['particle_number_per_grid_point'] = int(3 * dot_size_microns[i])
    
    # set a max size of 100 to avoid memory errors
    if(piv_simulation_parameters['bos_pattern']['particle_number_per_grid_point'] > 100):
        piv_simulation_parameters['bos_pattern']['particle_number_per_grid_point'] = 100
    '''
    piv_simulation_parameters['bos_pattern']['particle_number_per_grid_point'] = 100.


    # set the location where the rendered images will be saved
    piv_simulation_parameters['output_data']['bos_pattern_image_directory'] = \
        '/home/barracuda/a/lrajendr/Projects/camera_simulation/results/images/bos/error-analysis/dot-size/%dx%d-f%d-disp%d/%d' % (nx,ny,f_number,expected_displacement_pixels,dot_size_microns[i]
            ) + '/'

    print "output directory", piv_simulation_parameters['output_data']['bos_pattern_image_directory']

    # create the full filename where the parameters will be saved
    filename_full = filename_base + '%02d' % (i+1) + '.mat'

    # save parameters to file
    sio.savemat(filepath + filename_full, piv_simulation_parameters, appendmat=True, format='5', long_field_names=True)



