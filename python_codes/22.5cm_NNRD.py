'''
This program creates a 3D density field in NRRRD format for use in ray tracing.

The refractive index field is defined as n(x) = ax^2 + bx + c. 

'''

import nrrd
import numpy as np
import matplotlib.pyplot as plt
from sys import argv

def save_nrrd(data,nrrd_filename):
# This function saves the density stored in data['rho'] to an nrrd file
# specified by nrrd_filename

    # specify orientation of space
    space = '3D-right-handed'

    # generate arrays to store co-ordinates along the three axes
    x = np.array(data['x'])
    y = np.array(data['y'])
    z = np.array(data['z'])

    # set origin
    x0 = x.min()
    y0 = y.min()
    z0 = z.min()

    space_orig = np.array([x0, y0, z0]).astype('float32')

    # set grid spacing
    del_x = np.diff(x)[0]
    del_y = np.diff(y)[0]
    del_z = np.diff(z)[0]

    spacings = np.array([del_x, del_y, del_z]).astype('float32')

    # spcify other relevant options
    options = {'type' : 'f4', 'space': space, 'encoding': 'raw',
               'space origin' : space_orig, 'spacings' : spacings}

    print("saving density to %s \n" % nrrd_filename)

    # save data to nrrd file
    nrrd.write(nrrd_filename, np.array(data['rho']).astype('float32'), options)

#script, first, second, third = argv

data = {}
nx = 100
ny = 225
nz = 100
z_object = 0#750e3-5e4
#<<<<<<< Updated upstream

#Z_Min = -3.0e2
#Z_Max = 3.0e02

Z_Max = z_object - 10e4
Z_Min = z_object - 16e4
# X_Velocity = 01.0e3
# Y_Velocity = 01.0e3
#Z_Velocity = 00.1e3

data['x'] = np.linspace(-30e3,30e3,nx).astype('float32')# - X_Velocity/2
data['y'] = np.linspace(-11.25e4,2.25e4,ny).astype('float32')# - Y_Velocity/2
# data['z'] = np.linspace(-15e3,15e3,nz).astype('float32') + z_object - Z_Velocity/2
#data['z'] = np.linspace(8.7e5,9.5e5,nz).astype('float32')
#data['z'] = np.linspace(Z_Min,Z_Max,nz).astype('float32') + z_object
data['z'] = np.linspace(Z_Min, Z_Max, nz).astype('float32')

#print "del_x", data['x'][1] - data['x'][0]

'''
# convert co-ordinates from microns to meters
data['x']/=1.0e6
data['y']/=1.0e6
data['z']/=1.0e6
'''
# specify gradients for x and y directions
#grad_x = np.float64(first)

#grad_xx = 50.0 
grad_x = 5
grad_y = 0.0

x = np.array(data['x'])
y = np.array(data['y'])
z = np.array(data['z'])

X,Y = np.meshgrid(x,y,indexing='ij')

def n_syrup(T):
    return -2.058*1e-4*T +1.5012 -1

data['rho'] = n_syrup(25)*np.ones([nx,ny,nz])
g_mean = 0.5
g_amp = (n_syrup(25)+1)*0.0035#1.225*5e-1
X_eff = (X - x.min())/(X.max() - x.min())
g_std = 0.03

gz_mean = (Z_Max + Z_Min)/2
stem_radius = 0.15/2.5
gradient_width = 0.03

inner_radius = 0.35
outer_radius = 0.45
Y_interface = (y.min() + y.max())/2 + 2.95e4 + 0.5e4
def head_damp(radius,inner_radius,outer_radius):
    ridge_line = 0.8
    ridge_radius = (outer_radius - inner_radius) * ridge_line + inner_radius
    mask_upper = np.array(radius>=ridge_radius,dtype=int)
    mask_lower = np.array(radius<ridge_radius,dtype=int)
    damp_ = (radius - ridge_radius)/(outer_radius - ridge_radius)*mask_upper + (ridge_radius - radius)/(ridge_radius - inner_radius)*mask_lower
    return (1 - damp_)**2

def stem_damp(radius,stem_radius):
    platform_line = 0.9
    return (1-radius/stem_radius) * np.array(radius>=stem_radius*platform_line,dtype=int) + np.array(radius<stem_radius*platform_line,dtype=int)
for k in range(0,nz):
    rad_2 = (X_eff-g_mean)**2 + ((z[k]-gz_mean)/(X.max() - x.min()))**2
    stem_bool = (rad_2<=stem_radius**2)&(Y<=Y_interface+2e4)
    valley_mask = np.array(stem_bool,dtype=int)
    data['rho'][:,:,k] -= valley_mask*g_amp*0.9*np.exp(-rad_2/(2*g_std**2))#stem_damp(np.sqrt(rad_2),stem_radius)#np.exp(-rad_2/(2*g_std**2))

    rad_3d_2 = (X_eff-g_mean)**2 + ((z[k]-gz_mean)/(X.max() - x.min()))**2 + ((Y-Y_interface)/(X.max() - x.min()))**2
    shell_mask = np.array((rad_3d_2<=outer_radius**2)&(rad_3d_2>=inner_radius**2)&(np.invert(stem_bool))&(Y>Y_interface),dtype=int)
    data['rho'][:,:,k] -= shell_mask*g_amp*0.6*head_damp(np.sqrt(rad_3d_2),inner_radius,outer_radius)


    # gradient_mask = np.array(np.array(rad_2<radius**2)&np.array(rad_2>(radius-gradient_width)**2),dtype=int)
    # data['rho'][:,:,k] -= gradient_mask*g_amp/4
    # data['rho'][:,:,k] -= g_amp*np.exp(-(X_eff-g_mean)**2/(2*g_std**2))
# pattern = np.ones_like(X)*grad_x
# pattern[:len(x)//2]=0
# pattern[int(len(x)//2+len(x)//10):]=0
# for k in range(0,nz):
#     data['rho'][:,:,k] -= pattern * (X - x.min())/(X.max() - x.min()) #+ grad_xx * (X - x.min())**2/(X.max() - x.min())**2 # + grad_y*(Y - y.min())/Y.max()

#print "multiplying factor:", (X-x.min())/(X.max() - x.min())
#print "del_rho:", (data['rho'][1,0,0] - data['rho'][0,0,0])

#nrrd_filename = '/home/barracuda/a/lrajendr/Projects/ray_tracing_density_gradients/schlieren-0.2.0-Build/const_grad.nrrd'
#nrrd_filename = '/home/barracuda/a/lrajendr/Projects/parallel_ray_tracing/data/const_grad_BOS_grad_x_08.nrrd'
nrrd_filename = '../sample-data/piv/plume_1.nrrd'

save_nrrd(data,nrrd_filename)


# Create a figure with a grid of subplots
fig, axes = plt.subplots(2, 3, figsize=(15, 10))

# Plot the first graph
axes[0, 0].plot(data['x'], data['rho'][:, :, nz//2])
axes[0, 0].set_title("Density")
axes[0, 0].grid()

# Plot the second graph
im = axes[0, 1].imshow(data['rho'][:, :, nz//2])
axes[0, 1].set_xlabel("ymesh")
axes[0, 1].set_ylabel("xmesh")
fig.colorbar(im, ax=axes[0, 1])
axes[0, 1].set_title("XY")
axes[0, 1].grid()

# Plot the remaining four graphs
for i in range(4):
    im = axes[(i+2)//3, (i+2)%3].imshow(data['rho'][:, ny//4 * i, :])
    axes[(i+2)//3, (i+2)%3].set_xlabel("zmesh")
    axes[(i+2)//3, (i+2)%3].set_ylabel("xmesh")
    fig.colorbar(im, ax=axes[(i+2)//3, (i+2)%3])
    axes[(i+2)//3, (i+2)%3].set_title(f"Y{i+1}")
    axes[(i+2)//3, (i+2)%3].grid()

# Adjust layout
plt.tight_layout()
plt.show()

data_2 = data.copy()

data_2['rho'] = n_syrup(25)*np.ones([nx,ny,nz])
nrrd_filename = '../shadow_piv_sim/photon/sample-data/piv/plume_2.nrrd'
inner_radius = inner_radius*1.04
outer_radius = outer_radius*1.04
Y_interface = (y.min() + y.max())/2 + 3e4 + 0.5e4
for k in range(0,nz):
    rad_2 = (X_eff-g_mean+0.016)**2 + ((z[k]-gz_mean)/(X.max() - x.min()))**2
    stem_bool = (rad_2<=stem_radius**2)&(Y<=Y_interface+2e4)
    valley_mask = np.array(stem_bool,dtype=int)
    data_2['rho'][:,:,k] -= valley_mask*g_amp*0.9*np.exp(-rad_2/(2*g_std**2))#stem_damp(np.sqrt(rad_2),stem_radius)#np.exp(-rad_2/(2*g_std**2))

    rad_3d_2 = (X_eff-g_mean+0.016)**2 + ((z[k]-gz_mean)/(X.max() - x.min()))**2 + ((Y-Y_interface)/(X.max() - x.min()))**2
    shell_mask = np.array((rad_3d_2<=outer_radius**2)&(rad_3d_2>=inner_radius**2)&(np.invert(stem_bool))&(Y>Y_interface),dtype=int)
    data_2['rho'][:,:,k] -= shell_mask*g_amp*0.5*head_damp(np.sqrt(rad_3d_2),inner_radius,outer_radius)


save_nrrd(data_2,nrrd_filename)


# estimate ray deflection through the volume
del_rho = data['rho'][1,0,0] - data['rho'][0,0,0]
del_n = 0.226*1e-6/1e-3*del_rho
del_x = np.diff(data['x'])[0]

ray_deflection = del_n/del_x * abs(Z_Max - Z_Min)

print("del_x (microns) : %.2E, del_rho: %.2E, del_n: %.2E" % (del_x, del_rho, del_n))
print("theoretical deflection (radians) : %0.2E" % ray_deflection)

# Create a figure with a grid of subplots
fig, axes = plt.subplots(2, 1, figsize=(15, 10))

# Plot the second graph
plt.sca(axes[0])
im = plt.imshow(data['rho'][:, :, nz//2],vmin=0.49,vmax=0.496)
plt.xlabel("ymesh")
plt.ylabel("xmesh")
plt.colorbar(im)
plt.title("XY")
plt.grid()

plt.sca(axes[1])
im = plt.imshow(data_2['rho'][:, :, nz//2],vmin=0.49,vmax=0.496)
plt.xlabel("ymesh")
plt.ylabel("xmesh")
plt.colorbar(im)
plt.title("XY")
plt.grid()


# plt.figure()
# plt.plot(data['x'],data['rho'][:,:,nz//2])
# plt.title("density")
# plt.grid()
# plt.show()

# plt.figure()
# plt.imshow(data['rho'][:,:,nz//2])
# plt.xlabel("ymesh")
# plt.ylabel("xmesh")
# plt.colorbar()
# plt.title("xy")
# plt.grid()
# plt.show()

# for i in range(4):
#     plt.figure()
#     plt.imshow(data['rho'][:,ny//6*i,:])
#     plt.xlabel("zmesh")
#     plt.ylabel("xmesh")
#     plt.colorbar()
#     plt.title("y"+str(i+1))
#     plt.grid()
#     plt.show()

# Create a figure with a grid of subplots
fig, axes = plt.subplots(2, 3, figsize=(15, 10))

# Plot the first graph
axes[0, 0].plot(data['x'], data['rho'][:, :, nz//2])
axes[0, 0].set_title("Density")
axes[0, 0].grid()

# Plot the second graph
im = axes[0, 1].imshow(data['rho'][:, :, nz//2])
axes[0, 1].set_xlabel("ymesh")
axes[0, 1].set_ylabel("xmesh")
fig.colorbar(im, ax=axes[0, 1])
axes[0, 1].set_title("XY")
axes[0, 1].grid()

# Plot the remaining four graphs
for i in range(4):
    im = axes[(i+2)//3, (i+2)%3].imshow(data['rho'][:, ny//4 * i, :])
    axes[(i+2)//3, (i+2)%3].set_xlabel("zmesh")
    axes[(i+2)//3, (i+2)%3].set_ylabel("xmesh")
    fig.colorbar(im, ax=axes[(i+2)//3, (i+2)%3])
    axes[(i+2)//3, (i+2)%3].set_title(f"Y{i+1}")
    axes[(i+2)//3, (i+2)%3].grid()

# Adjust layout
plt.tight_layout()
plt.show()





'''
=======

X_Velocity = 01.0e3
Y_Velocity = 01.0e3
Z_Velocity = 00.1e3

data['x'] = np.linspace(-15e4,15e4,nx).astype('float32') - X_Velocity/2
data['y'] = np.linspace(-15e4,15e4,ny).astype('float32') - Y_Velocity/2
# data['z'] = np.linspace(-15e3,15e3,nz).astype('float32') + z_object - Z_Velocity/2
data['z'] = np.linspace(7.5e5,9.0e5,nz).astype('float32')
# specify gradients for x and y directions
grad_x = 5e-4
grad_y = 0.0

x = np.array(data['x'])
y = np.array(data['y'])
z = np.array(data['z'])

X,Y = np.meshgrid(x,y,indexing='ij')

data['rho'] = 1.225*np.ones([nx,ny,nz])
for k in range(0,nz):
    data['rho'][:,:,k] += grad_x*(X-x.min())/X.max() # + grad_y*(Y - y.min())/Y.max()

# data['rho'] = data['rho'].T
# plot density profile
# plt.plot(data['rho'][:,10,0])
# plt.show()
save_nrrd(data,'const_grad.nrrd')
>>>>>>> Stashed changes
'''




