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
from abacusnbody.analysis.power_spectrum import calc_pk_from_deltak
from obtain_IC_fields import compress_asdf
#from velocileptors.LPT.cleft_fftw import CLEFT # TESTING

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

    # get a few parameters for the simulation
    meta = get_meta(sim_name, redshift=0.5)
    Lbox = meta['BoxSize']
        
    # names of the fields
    fields = ['1cb', 'delta', 'delta2', 'nabla2', 'tidal2']
        
    # create save directory
    save_dir = Path(heft_dir) / sim_name
    pcle_dir = save_dir / "pcles"

    # file to save the advected fields
    adv_fields_fn = Path(save_dir) / f"adv_fields_nmesh{nmesh:d}.asdf"
    adv_power_fn = Path(save_dir) / f"adv_power_nmesh{nmesh:d}.asdf"
    
    # load advected fields
    adv_fields = asdf.open(adv_fields_fn)['data']

    # define k bins
    k_bin_edges = np.linspace(0, 1., 201) # h/Mpc
    mu_bin_edges = np.array([0., 1.]) # angle

    # define bin centers
    k_binc = (k_bin_edges[1:] + k_bin_edges[:-1])*.5
    mu_binc = (mu_bin_edges[1:] + mu_bin_edges[:-1])*.5
    
    # loop over fields # todo!!!! better to save field_fft
    adv_pk_dict = {}
    for i, field in enumerate(fields):
        # get the fft field
        # tuks
        if field == '1cb':
            field_fft = rfftn(adv_fields[field]/np.mean(adv_fields[field], dtype=np.float64) - np.float32(1.), workers=-1)/np.complex64(adv_fields[field].size)
        else:
            field_fft = rfftn(adv_fields[field]/np.mean(adv_fields['1cb'], dtype=np.float64), workers=-1)/np.complex64(adv_fields[field].size)
        
        for j, field2 in enumerate(fields):
            if i < j: continue
            
            # get the fft field2
            if field2 == "1cb":
                field2_fft = rfftn(adv_fields[field2]/np.mean(adv_fields[field2], dtype=np.float64) - np.float32(1.), workers=-1)/np.complex64(adv_fields[field2].size)
            else:
                field2_fft = rfftn(adv_fields[field2]/np.mean(adv_fields['1cb'], dtype=np.float64), workers=-1)/np.complex64(adv_fields[field2].size)
            
            # compute power spectrum
            adv_pk_dict[f'{field}_{field2}'] = calc_pk_from_deltak(field_fft, Lbox, k_bin_edges, mu_bin_edges, field2_fft=field2_fft)
            adv_pk_dict[f'{field}_{field2}']['k_min'] = k_bin_edges[:-1]
            adv_pk_dict[f'{field}_{field2}']['k_max'] = k_bin_edges[1:]
            adv_pk_dict[f'{field}_{field2}']['k_mid'] = k_binc
            adv_pk_dict[f'{field2}_{field}'] = adv_pk_dict[f'{field}_{field2}']
            
            del field2_fft
            gc.collect()

        del field_fft
        gc.collect()

    # save fields using asdf compression
    header = {}
    header['sim_name'] = sim_name
    header['Lbox'] = Lbox
    header['pcle_type'] = pcle_type # could be A or B
    header['paste'] = paste
    header['nmesh'] = nmesh
    header['kcut'] = kcut
    compress_asdf(str(adv_power_fn), adv_pk_dict, header)

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


