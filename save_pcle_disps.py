import argparse
import gc
import os
import glob
from pathlib import Path
import warnings

import asdf
import numpy as np
import yaml
import numba
#from np.fft import fftfreq, fftn, ifftn
from scipy.fft import rfftn, irfftn

from abacusnbody.data.compaso_halo_catalog import CompaSOHaloCatalog
from abacusnbody.metadata import get_meta

from obtain_IC_fields import compress_asdf

from asdf.exceptions import AsdfWarning
warnings.filterwarnings('ignore', category=AsdfWarning)

DEFAULTS = {'path2config': 'config/abacus_heft.yaml'}

def read_abacus(chunk_fn):
    """
    Read positions, velocities and IC positions
    """
    
    # load halo catalog and 3% particle subsample
    cat = CompaSOHaloCatalog(chunk_fn, subsamples=dict(A=True, rv=True, pid=True), fields=[], unpack_bits=True)

    # load the pid, position and lagrangian positions
    #pid_mat = cat.subsamples['pid']
    pos_mat = cat.subsamples['pos']
    vel_mat = cat.subsamples['vel']
    lagr_pos = cat.subsamples['lagr_pos']

    # halo catalog
    #halo_table = cat.halos # x_L2com, N
    #header = cat.header
    #N_halos = len(cat.halos)
    #print("N_halos = ",N_halos)

    return lagr_pos, pos_mat, vel_mat

def main(path2config, alt_simname=None, verbose=False):

    # read heft parameters
    config = yaml.safe_load(open(path2config))
    heft_dir = config['heft_params']['heft_dir']
    nmesh = config['heft_params']['nmesh']
    kcut = config['heft_params']['kcut']
    z_mock = config['sim_params']['z_mock']
    sim_dir = config['sim_params']['sim_dir']
    
    if alt_simname is not None:
        sim_name = alt_simname
    else:
        sim_name = config['sim_params']['sim_name']

    # create save directory
    save_dir = Path(heft_dir) / sim_name / "pcles"
    os.makedirs(save_dir, exist_ok=True)

    # get a few parameters for the simulation
    meta = get_meta(sim_name, redshift=z_mock)
    Lbox = meta['BoxSize']
    NP = meta['NP']
    # catalog directory
    cat_dir = Path(sim_dir) / sim_name / f"halos" / f"z{z_mock:.3f}" / "halo_info"
    chunk_fns = sorted(cat_dir.glob("halo_info_*.asdf")) #{i_chunk:03d}
    n_chunks = len(chunk_fns)
    print(n_chunks)
    
    """
    # estimate number of particles
    pcle_type = "A"
    if pcle_type == "A":
        N_total = NP*0.03

    # count number of particles
    N_total = 0
    for chunk_fn in chunk_fns:
        N_halo = len(asdf.open(chunk_fn)['data']['N'])
        N_total += N_halo

    N_total *= 1.05
    N_total = int(N_total)

    # fill with particles
    pos_pcles = np.empty((N_total, 3), dtype=np.float32)
    vel_pcles = np.empty((N_total, 3), dtype=np.float32)
    pos_lagrs = np.empty((N_total, 3), dtype=np.float32)
    """
    
    # loop over all chunks
    count = 0
    for i_chunk, chunk_fn in enumerate(chunk_fns):
        
            
        pos_lagr, pos_pcle, vel_pcle = read_abacus(chunk_fn)
        N_pcle = pos_lagr.shape[0]
        print(N_pcle)

        """
        pos_pcles[count: count+N_pcle] = pos_pcle
        vel_pcles[count: count+N_pcle] = vel_pcle
        pos_lagrs[count: count+N_pcle] = pos_lagr
        count += N_pcle
        """
        
        # save fields using asdf compression
        header = {}
        header['sim_name'] = sim_name
        header['Lbox'] = Lbox
        header['pcle_type'] = pcle_type # could be A or B
        table = {}
        table['pos_pcle'] = pos_pcle
        table['vel_pcle'] = vel_pcle
        table['pos_lagr'] = pos_lagr
        compress_asdf(str(pcle_fn), table, header)

        del pos_lagr, pos_pcle, vel_pcle
        gc.collect()
        
        """
        pos_pcles = pos_pcles[:count]
        vel_pcles = vel_pcles[:count]
        pos_lagrs = pos_lagrs[:count]
        """
    
class ArgParseFormatter(argparse.RawDescriptionHelpFormatter, argparse.ArgumentDefaultsHelpFormatter):
    pass

if __name__ == "__main__":

    # parsing arguments
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=ArgParseFormatter)
    parser.add_argument('--path2config', help='Path to the config file', default=DEFAULTS['path2config'])
    parser.add_argument('--alt_simname', help='Alternative simulation name')
    parser.add_argument('--verbose', action='store_true', help='Print out useful statements')
    args = vars(parser.parse_args())
    main(**args)
