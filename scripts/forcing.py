import subprocess
import numpy as np

import sys
import os
if os.getlogin() == 'robin':
    cluster_login = 'robinmid'
    subprocess.call(["bash", "/home/robin/scripts/mount_cluster_dirs.sh", "mount"])
    sys.path.append("/home/robin/repos/acclimate-system-response/scripts")
    sys.path.append("/home/robin/repos/post-processing/")
elif os.getlogin() == 'quante':
    cluster_login = 'quante'
    subprocess.call(["bash", "/home/quante/small_tools/mount_cluster_dirs.sh", "mount"])
    sys.path.append("/home/quante/git/acclimate-system-response/scripts")
    sys.path.append("/home/quante/git/post-processing/")

from utils import EORA_CHN_USA_REGIONS, write_ncdf_output, EORA_SECTORS
from acclimate.dataset import AcclimateOutput

local_cluster_project_dir = "/mnt/cluster/p/projects/compacts/projects/acclimate-system-response/"
local_template_dir = "/mnt/cluster/p/projects/compacts/projects/acclimate-system-response/acclimate-system-response/templates/"



def generate_dirac_impulse(regions, outpath, t_shock=5, magnitude=1, series_len=365):
    if type(regions) == str:
        regions = [regions]
    if sum([region not in EORA_CHN_USA_REGIONS for region in regions]) != 0:
        raise ValueError("One or more region in {} are not valid regions.".format(regions))
    if magnitude < 0 or magnitude > 1:
        raise ValueError("Magnitude must be in [0,1]")
    dirac_impulse = np.ones(series_len)
    dirac_impulse[t_shock] = 1 - magnitude
    forcing = {r: dirac_impulse for r in regions}
    write_ncdf_output(_forcing_curves=forcing, _sector_list=EORA_SECTORS, _out_file=outpath, max_len=series_len)


def generate_multi_day_dirac_impulse(regions, outpath, start_shock=5, end_shock=15, magnitude=1, series_len=365):
    if type(regions) == str:
        regions = [regions]
    if sum([region not in EORA_CHN_USA_REGIONS for region in regions]) != 0:
        raise ValueError("One or more region in {} are not valid regions.".format(regions))
    if magnitude < 0 or magnitude > 1:
        raise ValueError("Magnitude must be in [0,1]")
    dirac_impulse = np.ones(series_len)
    dirac_impulse[start_shock:end_shock] = 1 - magnitude
    forcing = {r: dirac_impulse for r in regions}
    write_ncdf_output(_forcing_curves=forcing, _sector_list=EORA_SECTORS, _out_file=outpath, max_len=series_len)


# generate a random, white noise style forcing with uniform distirbuted forcing between min_shock and max_shock, TODO: extend by alternative distributions
# defaults to randomly shocking all regions, but regions argument can be used to limit random forcing to the given regions

from numpy.random import default_rng


def generate_random_forcing(region_group, outpath, forcing_start_time=1, min_shock=0.0, max_shock=0.2, series_len=720, random_seed=None):

    if min_shock < 0 or min_shock > 1:
        raise ValueError("min_shock must be in [0,1]")
    if max_shock < 0 or max_shock > 1:
        raise ValueError("max_shock must be in [0,1]")
    # draw forcing for each region

    forcing = {}
    # seed rng if provided
    if random_seed == None:
        rng = default_rng()
    else:
        rng = default_rng(random_seed)

    for i_region in region_group:
        random_impulse = rng.uniform(min_shock, max_shock, series_len)
        # eunsure forcing free start of simulation in baseline state
        for i_timepoint in range(0,forcing_start_time,step=1):
            random_impulse[i_timepoint] = 0
        region_forcing = np.ones(series_len) - random_impulse
        forcing[i_region] = region_forcing
    write_ncdf_output(_forcing_curves=forcing, _sector_list=EORA_SECTORS, _out_file=outpath, max_len=series_len)

def generate_simulation_ensemble(name, region_groups=None, magnitudes=1, simulation_len=365, qos='medium',
                                 partition='standard', num_cpu=1, start_runs=False,
                                 forcing_generator=generate_dirac_impulse, *forcing_generator_args):
    mnt_path = '/home/quante/mnt/cluster'
    if os.getlogin() == 'quante':
        local_cluster_project_dir = "/home/quante/mnt/cluster/p/projects/compacts/projects/acclimate-system-response/"
        local_template_dir = "/home/quante/mnt/cluster/p/projects/compacts/projects/acclimate-system-response/acclimate-system-response/templates/"
        mnt_path = '/home/quante/mnt/cluster'
    
    ensemble_dir = os.path.join(local_cluster_project_dir, "acclimate_runs", name)
    
    if not os.path.exists(ensemble_dir):
        os.mkdir(ensemble_dir)

    if forcing_generator==generate_random_forcing:
        region_group = EORA_CHN_USA_REGIONS #TODO: enable choice of regions
        simulation_name = 'global' + '_random'
        simulation_dir = os.path.join(ensemble_dir, simulation_name)
        os.mkdir(simulation_dir)
        forcing_path = os.path.join(simulation_dir, 'forcing.nc')
        forcing_generator(region_group, forcing_path, series_len=simulation_len,
                          *forcing_generator_args)
        settings_target_path = os.path.join(simulation_dir, 'settings.yml')
        slurm_target_path = os.path.join(simulation_dir, 'slurm_script.sh')
        with open(os.path.join(local_template_dir, 'settings_template.yml'), "rt") as settings_file:
            settings_data = settings_file.read()
        settings_data = settings_data.replace('+++stop_time+++', str(simulation_len))
        settings_data = settings_data.replace('+++forcing_filepath+++', forcing_path.replace(mnt_path, ''))
        with open(settings_target_path, "wt") as settings_file:
            settings_file.write(settings_data)
        with open(os.path.join(local_template_dir, 'slurm_script_template.sh'), "rt") as slurm_file:
            slurm_data = slurm_file.read()
        slurm_data = slurm_data.replace('+++qos+++', qos)
        slurm_data = slurm_data.replace('+++partition+++', partition)
        slurm_data = slurm_data.replace('+++workdir+++', simulation_dir.replace(mnt_path, ''))
        slurm_data = slurm_data.replace('+++num_cpu+++', str(num_cpu))
        slurm_data = slurm_data.replace('+++job_name+++', 'acc-sys_{}'.format(simulation_name))
        with open(slurm_target_path, "wt") as slurm_file:
            slurm_file.write(slurm_data)
        if start_runs:
            command = "sbatch {}".format(os.path.join(simulation_dir.replace(mnt_path, ''), 'slurm_script.sh'))
            subprocess.call(["ssh", "{}@cluster.pik-potsdam.de".format(cluster_login), command])

    elif region_groups is None:
        region_groups = [[r] for r in EORA_CHN_USA_REGIONS]

        if type(magnitudes) is int:
            magnitudes = [magnitudes]
        for region_group in region_groups:
            for magnitude in magnitudes:
                simulation_name = '+'.join(region_group) + '_m{}'.format(magnitude)
                simulation_dir = os.path.join(ensemble_dir, simulation_name)
                os.mkdir(simulation_dir)
                forcing_path = os.path.join(simulation_dir, 'forcing.nc')
                forcing_generator(region_group, forcing_path, magnitude=magnitude, series_len=simulation_len,
                                       *forcing_generator_args)
                settings_target_path = os.path.join(simulation_dir, 'settings.yml')
                slurm_target_path = os.path.join(simulation_dir, 'slurm_script.sh')
                with open(os.path.join(local_template_dir, 'settings_template.yml'), "rt") as settings_file:
                    settings_data = settings_file.read()
                settings_data = settings_data.replace('+++stop_time+++', str(simulation_len))
                settings_data = settings_data.replace('+++forcing_filepath+++', forcing_path.replace(mnt_path, ''))
                with open(settings_target_path, "wt") as settings_file:
                    settings_file.write(settings_data)
                with open(os.path.join(local_template_dir, 'slurm_script_template.sh'), "rt") as slurm_file:
                    slurm_data = slurm_file.read()
                slurm_data = slurm_data.replace('+++qos+++', qos)
                slurm_data = slurm_data.replace('+++partition+++', partition)
                slurm_data = slurm_data.replace('+++workdir+++', simulation_dir.replace(mnt_path, ''))
                slurm_data = slurm_data.replace('+++num_cpu+++', str(num_cpu))
                slurm_data = slurm_data.replace('+++job_name+++', 'acc-sys_{}'.format(simulation_name))
                with open(slurm_target_path, "wt") as slurm_file:
                    slurm_file.write(slurm_data)
            if start_runs:
                command = "sbatch {}".format(os.path.join(simulation_dir.replace(mnt_path, ''), 'slurm_script.sh'))
                subprocess.call(["ssh", "{}@cluster.pik-potsdam.de".format(cluster_login), command])



