import numpy as np
import cIntegrator
import ForceFields
import os, time
import h5py
import logging

def genrandint():
    'Generates a random integer between 0 and (2^32)-1'
    x = 0
    for i in range(4):
        x = (x << 8)+ord(os.urandom(1))
    return x

def main():

    print('Setting up logging')
    logging.basicConfig(filename='sim.log',level=logging.DEBUG)

    print('Setting consts')
    NUM_BLOCKS = int(1E4)
    STEPS_PER_BLOCK = 100
    BLOCKS_PER_DUMP = 1000
    ff = ForceFields.MuellerForce()
    
    MASS = 1.0
    FRICTION = 100.0
    BETA = 0.05 
    NDIMS = 2
    DT = 0.0001
    
    print('Instantiating Integrator')
    integrator = cIntegrator.Integrator(ff,MASS,FRICTION,BETA,DT,NDIMS,genrandint())

    # Setup hdf5 for storage
    print('Setting up HDF5')
    f = h5py.File('traj.h5','w')
    coords = f.create_dataset('coords',(NUM_BLOCKS,NDIMS),compression='gzip')
    
    # Allocate numpy storage for temp storage of positions
    ctemp = np.zeros((BLOCKS_PER_DUMP,2))    

    # Initial coords and velocities
    x = np.array([0.5,0.0])
    v = np.array([0.0,0.0])
    
    totblocks = NUM_BLOCKS//BLOCKS_PER_DUMP
    print('Starting Simulation')
    for dk in xrange(totblocks):
        t1 = time.time()
        for k in xrange(BLOCKS_PER_DUMP):
            integrator.step(x,v,STEPS_PER_BLOCK)
            ctemp[k,:] = x
            
        coords[dk*BLOCKS_PER_DUMP:(dk+1)*BLOCKS_PER_DUMP,:] = ctemp
        logging.info('Completed {} of {} steps: {} s'.format(dk,totblocks-1,time.time() - t1))




    f.close()


if __name__ == '__main__':
    main()
