import argparse
import gc
import os
from pathlib import Path
import warnings

import asdf
import numpy as np
import yaml
import numba
#from np.fft import fftfreq, fftn, ifftn
from scipy.fft import rfftn, irfftn

from classy import Class
from abacusnbody.metadata import get_meta
from abacusnbody.analysis.tsc import tsc_parallel
from obtain_IC_fields import compress_asdf


from asdf.exceptions import AsdfWarning
warnings.filterwarnings('ignore', category=AsdfWarning)

DEFAULTS = {'path2config': 'config/abacus_heft.yaml'}

def main(path2config, alt_simname=None, verbose=False):
    r"""
    Save the advacted fields (1cb, delta, delta^2, s^2, nabla^2) as ASDF files.

    Parameters
    ----------
    path2config : str
        name of the yaml containing parameter specifications.
    alt_simname : str, optional
        specify simulation name if different from yaml file.
    verbose : bool, optional
        print some useful benchmark statements. Default is False.
    """
    
    # read heft parameters
    config = yaml.safe_load(open(path2config))
    heft_dir = config['heft_params']['heft_dir']
    ic_dir = config['heft_params']['ic_dir']
    nmesh = config['heft_params']['nmesh']
    kcut = config['heft_params']['kcut']
    z_mock = config['sim_params']['z_mock']
    #sim_dir = config['sim_params']['sim_dir']
    paste = "TSC"
    pcle_type = "A"
    
    if alt_simname is not None:
        sim_name = alt_simname
    else:
        sim_name = config['sim_params']['sim_name']

    # names of the fields
    factors_fields = {'delta': 1., 'delta2': 2., 'nabla2': 1., 'tidal2': 2}
        
    # create save directory
    save_dir = Path(heft_dir) / sim_name
    pcle_dir = save_dir / "pcles"

    # all particle files
    pcle_fns = sorted(pcle_dir.glob(f"pcle_info_*.asdf"))
    n_chunks = len(pcle_fns)
    print(n_chunks)

    # get a few parameters for the simulation
    meta = get_meta(sim_name, redshift=0.5)
    Lbox = meta['BoxSize']
    z_ic = meta['InitialRedshift']
    D_growth = meta['GrowthTable'][z_mock]/meta['GrowthTable'][z_ic]

    # file to save the advected fields
    adv_fields_fn = Path(save_dir) / f"adv_fields_nmesh{nmesh:d}.asdf"
    
    # load fields
    fields_fn = Path(save_dir) / f"fields_nmesh{nmesh:d}.asdf"
    f = asdf.open(fields_fn)

    # initialize advected fields
    adv_fields = {}
    for field in factors_fields.keys():
        adv_fields[field] = np.zeros((nmesh, nmesh, nmesh), dtype=np.float32)
    adv_fields['1cb'] = np.zeros((nmesh, nmesh, nmesh), dtype=np.float32)
        
    # loop over all files
    for i_chunk, pcle_fn in enumerate(pcle_fns):
        print(i_chunk)
        
        # load particle in this chunk
        data = asdf.open(pcle_fn)['data']
        lagr_pos = data['pos_lagr']
        pcle_pos = data['pos_pcle']

        # get i, j, k for position on the density array
        lagr_ijk = ((lagr_pos+Lbox/2.)/(Lbox/nmesh)).astype(int)%nmesh
        del lagr_pos

        # loop over fields
        for field in factors_fields.keys():
            # get weight
            w = (f['data'][field]*D_growth**factors_fields[field])[lagr_ijk[:,0], lagr_ijk[:,1], lagr_ijk[:,2]]

            # add to field
            if paste == "TSC":
                tsc_parallel(pcle_pos, adv_fields[field], Lbox, weights=w)
            del w
            gc.collect()
            
        # add to field
        if paste == "TSC":
            tsc_parallel(pcle_pos, adv_fields['1cb'], Lbox, weights=None)

        # TODO! compensated and interlaced
        del pcle_pos
        gc.collect()
        
    # save fields using asdf compression
    header = {}
    header['sim_name'] = sim_name
    header['Lbox'] = Lbox
    header['pcle_type'] = pcle_type # could be A or B
    header['paste'] = paste
    header['nmesh'] = nmesh
    header['kcut'] = kcut
    compress_asdf(str(adv_fields_fn), adv_fields, header)
        
    # not needed but good to have
    """
    # define cosmology
    cosmo = {}
    cosmo['output'] = 'mPk mTk'
    cosmo['P_k_max_h/Mpc'] = 20.
    for k in ('H0', 'omega_b', 'omega_cdm',
              'omega_ncdm', 'N_ncdm', 'N_ur',
              'n_s', 'A_s', 'alpha_s',
              #'wa', 'w0',
    ):
        cosmo[k] = meta[k]

    # load input linear power
    kth = meta['CLASS_power_spectrum']['k (h/Mpc)']
    pk_z1 = meta['CLASS_power_spectrum']['P (Mpc/h)^3']

    # compute growth factor
    pkclass = Class()
    pkclass.set(cosmo)
    pkclass.compute()
    D_ratio = pkclass.scale_independent_growth_factor(z_mock)
    D_ratio /= pkclass.scale_independent_growth_factor(z_ic)

    # rewind back to initial redshift of the simulation
    p_m_lin = D_ratio**2*pk_z1

    # apply gaussian cutoff to linear power
    p_m_lin *= np.exp(-(kth/kcut)**2)
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


