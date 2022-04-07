import os
import subprocess

from .utils import EORA_CHN_USA_REGIONS, write_ncdf_output, EORA_SECTORS
import numpy as np

local_cluster_project_path = "/mnt/cluster/p/projects/compacts/projects/acclimate-system-response/"
local_template_dir = "/mnt/cluster/p/projects/compacts/projects/acclimate-system-response/templates/"

if os.getlogin() == 'robin':
    cluster_login = 'robinmid'
elif os.getlogin() == 'lennart':
    cluster_login = 'lquante'

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


def generate_simulation_ensemble(name, region_groups=None, magnitudes=1, simulation_len=365, qos='medium',
                                 partition='standard', num_cpu=1, start_runs=False,
                                 forcing_generator=generate_dirac_impulse, *forcing_generator_args):
    ensemble_dir = os.path.join(local_cluster_project_path, name)
    if not os.path.exists(ensemble_dir):
        os.mkdir(ensemble_dir)
    if region_groups is None:
        region_groups = [[r] for r in EORA_CHN_USA_REGIONS]
    if type(magnitudes) is int:
        magnitudes = [magnitudes]
    for region_group in region_groups:
        for magnitude in magnitudes:
            simulation_name = '+'.join(region_group) + '_m{}'.format(magnitude)
            simulation_dir = os.path.join(ensemble_dir, simulation_name)
            forcing_path = os.path.join(simulation_dir, simulation_name, 'forcing.nc')
            generate_dirac_impulse(region_group, forcing_path, magnitude=magnitude,
                                   series_len=simulation_len, *forcing_generator_args)
            settings_target_path = os.path.join(simulation_dir, 'settings.yml')
            slurm_target_path = os.path.join(simulation_dir, 'slurm_script.sh')
            subprocess.call(["cp", os.path.join(local_template_dir, 'settings_template.yml'), settings_target_path])
            subprocess.call(["cp", os.path.join(local_template_dir, 'slurm_script_template.sh'), slurm_target_path])
            with open(settings_target_path, "rt") as settings_file:
                settings_data = settings_file.read()
                settings_data.replace('+++stop_time+++', str(simulation_len))
                settings_data.replace('+++forcing_filepath+++', forcing_path.replace('/mnt/cluster', ''))
                settings_file.close()
            with open(settings_target_path, "wt") as settings_file:
                settings_file.write(settings_data)
            with open(slurm_target_path, "rt") as slurm_file:
                slurm_data = slurm_file.read()
                slurm_data.replace('+++qos+++', qos)
                slurm_data.replace('+++partition+++', partition)
                slurm_data.replace('+++workdir+++', simulation_dir.replace('/mnt/cluster', ''))
                slurm_data.replace('+++num_cpu+++', str(num_cpu))
                slurm_data.replace('+++job_name+++', 'acc-sys_{}'.format(simulation_name))
                with open(slurm_target_path, "wt") as slurm_file:
                    slurm_file.write(slurm_data)
        if start_runs:
            command = "sbatch {}".format(os.path.join(simulation_dir.replace('/mnt/cluster', ''), 'slurm_script.sh'))
            subprocess.call(["ssh", "{}@cluster.pik-potsdam.de".format(cluster_login), command])
