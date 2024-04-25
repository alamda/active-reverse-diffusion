from .context import read_configs

from read_configs import Configs

import pathlib

class ConfigsTest_Factory:
    filename = "test.tmp"

    sample_size = 100
    sample_dim = 2
    xmin = -2.0
    xmax = 2.0
    ymin = -1.0
    ymax = 1.0
    num_hist_bins = 20

    diffusion_calculation_type = 'numeric'
    num_diffusion_steps = 123
    dt = 0.04
    passive_training_iters = 100
    active_training_iters = 200
    reverse_sample_step_interval = 10

    passive_noise_T = 0.2

    active_noise_Tp = 0.1
    active_noise_Ta = 5
    active_noise_tau = 0.25

    target_type = 'gaussian'

    mu_x_list = [-0.5, 0.3]
    mu_y_list = [0, 0]
    sigma_list = [0.1, 0.1]
    pi_list = [0.3, 0.7]
    
    ofile_base = "test_ofile_base"
    
    forward_passive_fname = "forward_passive.npy"
    forward_active_x_fname = "forward_active_x.npy"
    forward_active_eta_fname = "forward_active_eta.npy"
    
    target_fname = "target.npy"
    
    models_passive_fname = "models_passive.pkl"
    models_active_x_fname = "models_active_x.pkl"
    models_active_eta_fname = "models_active_eta.pkl"

    def write_config_file(self):
        with open(self.filename, 'w') as f:
            newline = '\n'

            f.write(f'[sample]{newline}')
            f.write(f'size: {self.sample_size}{newline}')
            f.write(f'dim: {self.sample_dim}{newline}')
            f.write(f'xmin: {self.xmin}{newline}')
            f.write(f'xmax: {self.xmax}{newline}')
            f.write(f'ymin: {self.ymin}{newline}')
            f.write(f'ymax: {self.ymax}{newline}')
            f.write(f'num_hist_bins: {self.num_hist_bins}{newline}')
            f.write(f'{newline}')

            f.write(f'[diffusion]{newline}')
            f.write(f'calc_type: {self.diffusion_calculation_type}{newline}')
            f.write(f'num_steps: {self.num_diffusion_steps}{newline}')
            f.write(f'dt: {self.dt}{newline}')
            f.write(f'passive_training_iters: {self.passive_training_iters}{newline}')
            f.write(f'active_training_iters: {self.active_training_iters}{newline}')
            f.write(f'reverse_sample_step_interval: {self.reverse_sample_step_interval}{newline}')
            f.write(f'{newline}')

            f.write(f'[passive noise]{newline}')
            f.write(f'T: {self.passive_noise_T}{newline}')
            f.write(f'{newline}')

            f.write(f'[active noise]{newline}')
            f.write(f'Tp: {self.active_noise_Tp}{newline}')
            f.write(f'Ta: {self.active_noise_Ta}{newline}')
            f.write(f'tau: {self.active_noise_tau}{newline}')
            f.write(f'{newline}')

            mu_x_string = ''
            for mu in self.mu_x_list:
                mu_x_string += f'{mu}, '
            mu_x_string = mu_x_string[:-2]
            
            mu_y_string = ''
            for mu in self.mu_y_list:
                mu_y_string += f'{mu}, '
            mu_y_string = mu_y_string[:-2]

            sigma_string = ''
            for sigma in self.sigma_list:
                sigma_string += f'{sigma}, '
            sigma_string = sigma_string[:-2]

            pi_string = ''
            for pi in self.pi_list:
                pi_string += f'{pi}, '
            pi_string = pi_string[:-2]

            f.write(f'[target]{newline}')
            f.write(f'type: {self.target_type}{newline}')
            f.write(f'mu_x_list: {mu_x_string}{newline}')
            f.write(f'mu_y_list: {mu_y_string}{newline}')
            f.write(f'sigma_list: {sigma_string}{newline}')
            f.write(f'pi_list: {pi_string}{newline}')
            f.write(f'{newline}')
            
            f.write(f'[output]{newline}')
            f.write(f'file_name: {self.ofile_base}{newline}')
            f.write(f'{newline}')
            
            f.write(f'[file names]{newline}')
            f.write(f'forward_passive: {self.forward_passive_fname}{newline}')
            f.write(f'forward_active_x: {self.forward_active_x_fname}{newline}')
            f.write(f'forward_active_eta: {self.forward_active_eta_fname}{newline}')
            f.write(f'target: {self.target_fname}{newline}')
            f.write(f'models_passive: {self.models_passive_fname}{newline}')
            f.write(f'models_active_x: {self.models_active_x_fname}{newline}')
            f.write(f'models_active_eta: {self.models_active_eta_fname}{newline}')

    def delete_config_file(self):
        file_path = pathlib.Path(self.filename)
        file_path.unlink()


def test_init():
    myFactory = ConfigsTest_Factory()
    myFactory.write_config_file()

    myConfigs = Configs(filename=myFactory.filename)

    assert myConfigs.sample_size == myFactory.sample_size
    assert myConfigs.sample_dim == myFactory.sample_dim
    assert myConfigs.xmin == myFactory.xmin
    assert myConfigs.xmax == myFactory.xmax
    assert myConfigs.ymin == myFactory.ymin
    assert myConfigs.ymax == myFactory.ymax
    assert myConfigs.num_hist_bins == myFactory.num_hist_bins

    assert myConfigs.diffusion_calculation_type == myFactory.diffusion_calculation_type
    assert myConfigs.num_diffusion_steps == myFactory.num_diffusion_steps
    assert myConfigs.dt == myFactory.dt
    assert myConfigs.passive_training_iters == myFactory.passive_training_iters
    assert myConfigs.active_training_iters == myFactory.active_training_iters

    assert myConfigs.passive_noise_T == myFactory.passive_noise_T

    assert myConfigs.active_noise_Tp == myFactory.active_noise_Tp
    assert myConfigs.active_noise_Ta == myFactory.active_noise_Ta
    assert myConfigs.active_noise_tau == myFactory.active_noise_tau

    assert myConfigs.target_type == myFactory.target_type
    assert myConfigs.mu_x_list == myFactory.mu_x_list
    assert myConfigs.mu_y_list == myFactory.mu_y_list
    assert myConfigs.sigma_list == myFactory.sigma_list
    assert myConfigs.pi_list == myFactory.pi_list
    
    assert myConfigs.ofile_base == myFactory.ofile_base
    
    assert myConfigs.forward_passive_fname == myFactory.forward_passive_fname
    assert myConfigs.forward_active_x_fname == myFactory.forward_active_x_fname
    assert myConfigs.forward_active_eta_fname == myFactory.forward_active_eta_fname
    assert myConfigs.target_fname == myFactory.target_fname
    assert myConfigs.models_passive_fname == myFactory.models_passive_fname
    assert myConfigs.models_active_x_fname == myFactory.models_active_x_fname
    assert myConfigs.models_active_eta_fname == myFactory.models_active_eta_fname

    myFactory.delete_config_file()
