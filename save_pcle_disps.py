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
from abacusnbody.data.bitpacked import unpack_pids, unpack_rvint

from obtain_IC_fields import compress_asdf

from asdf.exceptions import AsdfWarning
warnings.filterwarnings('ignore', category=AsdfWarning)

DEFAULTS = {'path2config': 'config/abacus_heft.yaml'}

def read_abacus_halo(chunk_fn):
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

def read_abacus(rv_fn, pid_fn, Lbox, PPD):

    # read the halo (L0+L1) matter particles
    rv_data = asdf.open(rv_fn)['data']['rvint'][:]
    pid_data = asdf.open(pid_fn)['data']['packedpid'][:]

    # unpack positions
    pos_mat, vel_mat = unpack_rvint(rv_data, Lbox, float_dtype=np.float32, velout=None)
    lagr_pos = unpack_pids(pid_data, box=Lbox, ppd=PPD, lagr_pos=True)['lagr_pos']
    del rv_data, pid_data
    gc.collect()

    return lagr_pos, pos_mat, vel_mat

def main(path2config, alt_simname=None, verbose=False):

    # read heft parameters
    config = yaml.safe_load(open(path2config))
    heft_dir = config['heft_params']['heft_dir']
    nmesh = config['heft_params']['nmesh']
    kcut = config['heft_params']['kcut']
    z_mock = config['sim_params']['z_mock']
    sim_dir = config['sim_params']['sim_dir']
    pcle_type = "A"
    
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
    PPD = int(NP**(1./3))
    
    # catalog directory
    cat_dir = Path(sim_dir) / sim_name / f"halos" / f"z{z_mock:.3f}" / "halo_info"
    chunk_fns = sorted(cat_dir.glob("halo_info_*.asdf")) #{i_chunk:03d}
    n_chunks = len(chunk_fns)
    print(n_chunks)
    
    """
    # estimate number of particles
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
        print(i_chunk)
            
        #pos_lagr, pos_pcle, vel_pcle = read_abacus_halo(chunk_fn) # only halo particles
        
        # halo and field particle files
        # TODO make pretty
        halo_rv_fn = sim_dir+f'/{sim_name}/halos/z{z_mock:.3f}/halo_rv_{pcle_type}/halo_rv_{pcle_type}_{i_chunk:03d}.asdf'
        field_rv_fn = sim_dir+f'/{sim_name}/halos/z{z_mock:.3f}/field_rv_{pcle_type}/field_rv_{pcle_type}_{i_chunk:03d}.asdf'
        halo_pid_fn = sim_dir+f'/{sim_name}/halos/z{z_mock:.3f}/halo_pid_{pcle_type}/halo_pid_{pcle_type}_{i_chunk:03d}.asdf'
        field_pid_fn = sim_dir+f'/{sim_name}/halos/z{z_mock:.3f}/field_pid_{pcle_type}/field_pid_{pcle_type}_{i_chunk:03d}.asdf'

        # load halo and field particles
        halo_lagr_pos, halo_pos_mat, halo_vel_mat = read_abacus(halo_rv_fn, halo_pid_fn, Lbox, PPD)
        field_lagr_pos, field_pos_mat, field_vel_mat = read_abacus(field_rv_fn, field_pid_fn, Lbox, PPD)
        
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
        table['halo_pos_pcle'] = halo_pos_mat
        table['halo_vel_pcle'] = halo_vel_mat
        table['halo_pos_lagr'] = halo_lagr_pos
        table['field_pos_pcle'] = field_pos_mat
        table['field_vel_pcle'] = field_vel_mat
        table['field_pos_lagr'] = field_lagr_pos
        pcle_fn = save_dir / f"pcle_info_{i_chunk:03d}.asdf"
        compress_asdf(str(pcle_fn), table, header)

        del halo_lagr_pos, halo_pos_mat, halo_vel_mat
        del field_lagr_pos, field_pos_mat, field_vel_mat
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
