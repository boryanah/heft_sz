import argparse
import gc
import os
import sys
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
sys.path.append('/global/homes/b/boryanah/repos/velocileptors')
from velocileptors.LPT.cleft_fftw import CLEFT

from asdf.exceptions import AsdfWarning
warnings.filterwarnings('ignore', category=AsdfWarning)

DEFAULTS = {'path2config': 'config/abacus_heft.yaml'}

# matplotlib settings
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt
"""
from matplotlib import rc
#rc('font',**{'family':'sans-serif','sans-serif':['Helvetica']})
rc('font',**{'family':'serif','serif':['Times'],'size':18})
rc('text', usetex=True)
"""
hexcols = np.array(['#44AA99', '#117733', '#999933', '#88CCEE', '#332288', '#BBBBBB', '#4477AA',
                    '#CC6677', '#AA4499', '#6699CC', '#AA4466', '#882255', '#661100', '#0099BB', '#DDCC77'])

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
    z_ic = meta['InitialRedshift']
    D_z1 = meta['GrowthTable'][1.0]
    D_mock = meta['GrowthTable'][z_mock]
    D_ic = meta['GrowthTable'][z_ic]
        
    # names of the fields
    fields = ['1cb', 'delta', 'delta2', 'nabla2', 'tidal2']
        
    # create save directory
    save_dir = Path(heft_dir) / sim_name

    # file to save the advected fields
    adv_power_fn = Path(save_dir) / f"adv_power_nmesh{nmesh:d}.asdf"

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
    
    # compute growth factor
    pkclass = Class()
    pkclass.set(cosmo)
    pkclass.compute()
    D_ratio = pkclass.scale_independent_growth_factor(z_mock)
    D_ratio /= pkclass.scale_independent_growth_factor(z_ic)
    """

    # load input linear power
    kth = meta['CLASS_power_spectrum']['k (h/Mpc)']
    pk_z1 = meta['CLASS_power_spectrum']['P (Mpc/h)^3']
    
    # rewind back to interest redshift of the simulation
    pk_cb = (D_mock/D_z1)**2*pk_z1
    
    # apply gaussian cutoff to linear power
    if not np.isclose(kcut, 0.):
        pk_cb *= np.exp(-(kth/kcut)**2)
    
    # Initialize the class -- with no wisdom file passed it will
    # experiment to find the fastest FFT algorithm for the system.
    # B.H. modified velocileptors/Utils/loginterp.py in case of exception error
    cleft = CLEFT(kth, pk_cb, cutoff=10)
    # You could save the wisdom file here if you wanted:
    # mome.export_wisdom(wisdom_file_name)

    # The first four are deterministic Lagrangian bias up to third order
    # While alpha and sn are the counterterm and stochastic term (shot noise)
    cleft.make_ptable()
    kv = cleft.pktable[:, 0]

    # parse velocileptors
    pk_theo = {'1cb_1cb': cleft.pktable[:, 1],\
               '1cb_delta': 0.5*cleft.pktable[:, 2], 'delta_delta': cleft.pktable[:, 3],\
               '1cb_delta2': 0.5*cleft.pktable[:, 4], 'delta_delta2': 0.5*cleft.pktable[:, 5],  'delta2_delta2': cleft.pktable[:, 6],\
               '1cb_tidal2': 0.5*cleft.pktable[:, 7], 'delta_tidal2': 0.5*cleft.pktable[:, 8],  'delta2_tidal2': 0.5*cleft.pktable[:, 9],\
               'tidal2_tidal2': cleft.pktable[:, 10], '1cb_nabla2': kv**2*0.5*cleft.pktable[:, 2],\
               'delta_nabla2': kv**2*cleft.pktable[:, 3], 'nabla2_nabla2': np.interp(kv, kth, pk_cb*kth**2),
               'nabla2_tidal2': kv**2*0.5*cleft.pktable[:, 8], 'delta2_nabla2': kv**2*0.5*cleft.pktable[:, 5]}
    """
               'delta_nabla2': np.interp(kv, kth, pk_cb*kth**2), 'nabla2_nabla2': np.interp(kv, kth, pk_cb*kth**2),
               'nabla2_tidal2': np.interp(kv, kth, pk_cb*kth**2), 'delta2_nabla2': np.interp(kv, kth, pk_cb*kth**2)}
    """
    # just so we don't have to worry
    pk_theo_copy = {} # lazy
    for key in pk_theo.keys():
        pk_theo_copy[key] = pk_theo[key]
    for pk in pk_theo_copy.keys():
        pk_theo[f"{pk.split('_')[1]}_{pk.split('_')[0]}"] = pk_theo[pk]

    # load power
    pk_data = asdf.open(adv_power_fn)['data']

    # plot result
    plt.subplots(2, 3, figsize=(18,10))
    count = 0
    for i, field in enumerate(fields):
        for j, field2 in enumerate(fields):
            if i < j: continue
            #plt.subplot(2, 3, count % 6 + 1)
            # TODO make pretty just to make comparison easier
            if "1cb_1cb" == f"{field}_{field2}": plt.subplot(2, 3, 1)
            if ("1cb_delta" == f"{field}_{field2}") or ("1cb_delta" == f"{field2}_{field}"): plt.subplot(2, 3, 1)
            if "delta_delta" == f"{field}_{field2}": plt.subplot(2, 3, 2)
            if ("delta_delta2" == f"{field}_{field2}") or ("delta_delta2" == f"{field2}_{field}"): plt.subplot(2, 3, 2)
            if "delta2_delta2" == f"{field}_{field2}": plt.subplot(2, 3, 3)
            if ("delta2_tidal2" == f"{field}_{field2}") or ("delta2_tidal2" == f"{field2}_{field}"): plt.subplot(2, 3, 3)
            if ("delta2_nabla2" == f"{field}_{field2}") or ("delta2_nabla2" == f"{field2}_{field}"): plt.subplot(2, 3, 3)
            if ("1cb_delta2" == f"{field}_{field2}") or ("1cb_delta2" == f"{field2}_{field}"): plt.subplot(2, 3, 4)
            if ("1cb_tidal2" == f"{field}_{field2}") or ("1cb_tidal2" == f"{field2}_{field}"): plt.subplot(2, 3, 4)
            if ("1cb_nabla2" == f"{field}_{field2}") or ("1cb_nabla2" == f"{field2}_{field}"): plt.subplot(2, 3, 4)
            if ("delta_tidal2" == f"{field}_{field2}") or ("delta_tidal2" == f"{field2}_{field}"): plt.subplot(2, 3, 5)
            if ("delta_nabla2" == f"{field}_{field2}") or ("delta_nabla2" == f"{field2}_{field}"): plt.subplot(2, 3, 5)
            if "nabla2_nabla2" == f"{field}_{field2}": plt.subplot(2, 3, 6)
            if "tidal2_tidal2" == f"{field}_{field2}": plt.subplot(2, 3, 6)
            if ("tidal2_nabla2" == f"{field}_{field2}") or ("tidal2_nabla2" == f"{field2}_{field}"): plt.subplot(2, 3, 6)
            
            k_mid = pk_data[f"{field}_{field2}"]['k_mid'].flatten()
            Pk_data = pk_data[f"{field}_{field2}"]['power'].flatten()
            Pk_theo = pk_theo[f"{field}_{field2}"]
            
            # LPT defines 1/2 (delta^2-<delta^2>)
            if 'delta2' in field:
                Pk_data /= 2.
            if 'delta2' in field2:
                Pk_data /= 2.

            # those are negative so we make them positive in order to show them in logpsace
            if ((field == 'delta' or field == '1cb') and field2 == 'tidal2') or ((field2 == 'delta' or field2 == '1cb') and field == 'tidal2'):
                Pk_data *= -1 
                Pk_theo *= -1
                
            # this term is positive if nabla^2 delta = -k^2 delta, but reason we multiply here is that we use k^2 delta instead and k^2 P_zeldovich
            """
            if (field == 'nabla2' and field2 == 'tidal2') or (field2 == 'nabla2' and field == 'tidal2'):
                Pk_data *= -1
            """
            D = 1.#51.77
            if "delta" == field and "delta" == field2:
                Pk_data /= D**2
            elif "delta" == field or "delta" == field2:
                Pk_data /= D
            if "delta2" == field and "delta2" == field2:
                Pk_data /= D**2
            elif "delta2" == field or "delta2" == field2:
                Pk_data /= D
            if "nabla2" == field and "nabla2" == field2:
                Pk_data /= D**2
            elif "nabla2" == field or "nabla2" == field2:
                Pk_data /= D
            if "tidal2" == field and "tidal2" == field2:
                Pk_data /= D**2
            elif "tidal2" == field or "tidal2" == field2:
                Pk_data /= D
            
            
            plt.plot(kv, np.abs(Pk_theo), ls='--', color=hexcols[count]) # s del
            plt.plot(k_mid, np.abs(Pk_data), color=hexcols[count], label=f"{field},{field2}")

            plt.xscale('log')
            plt.yscale('log')
            plt.legend(frameon=False, loc='best')
            count += 1
            
    plt.savefig("figs/comparison.png")
            
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


