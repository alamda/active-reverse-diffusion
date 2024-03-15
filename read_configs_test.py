from read_configs import Configs

import pathlib


class ConfigsTest_Factory:
    filename = "test"

    ofile_base = "test_ofile_base"

    sample_dim = 100
    xmin = -2.0
    xmax = 2.0
    num_hist_bins = 20

    num_diffusion_steps = 123
    dt = 0.04

    passive_noise_T = 0.2
    passive_training_iterations = 500

    active_noise_Tp = 0.1
    active_noise_Ta = 5
    active_noise_tau = 0.25
    active_training_iterations = 1000

    target_type = 'gaussian'

    mu_list = [-0.5, 0.3]
    sigma_list = [0.1, 0.1]
    pi_list = [0.3, 0.7]

    def write_config_file(self):
        with open(self.filename, 'w') as f:
            newline = '\n'
            f.write(f'[output]{newline}')
            f.write(f'file_name: {self.ofile_base}{newline}')
            f.write(f'{newline}')

            f.write(f'[sample]{newline}')
            f.write(f'dimension: {self.sample_dim}{newline}')
            f.write(f'xmin: {self.xmin}{newline}')
            f.write(f'xmax: {self.xmax}{newline}')
            f.write(f'num_hist_bins: {self.num_hist_bins}{newline}')
            f.write(f'{newline}')

            f.write(f'[diffusion]{newline}')
            f.write(f'num_steps: {self.num_diffusion_steps}{newline}')
            f.write(f'dt: {self.dt}{newline}')
            f.write(f'{newline}')

            f.write(f'[passive noise]{newline}')
            f.write(f'T: {self.passive_noise_T}{newline}')
            f.write(
                f'training_iterations: {self.passive_training_iterations}{newline}')
            f.write(f'{newline}')

            f.write(f'[active noise]{newline}')
            f.write(f'Tp: {self.active_noise_Tp}{newline}')
            f.write(f'Ta: {self.active_noise_Ta}{newline}')
            f.write(f'tau: {self.active_noise_tau}{newline}')
            f.write(
                f'training_iterations: {self.active_training_iterations}{newline}')
            f.write(f'{newline}')

            mu_string = ''
            for mu in self.mu_list:
                mu_string += f'{mu}, '
            mu_string = mu_string[:-2]

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
            f.write(f'mu_list: {mu_string}{newline}')
            f.write(f'sigma_list: {sigma_string}{newline}')
            f.write(f'pi_list: {pi_string}{newline}')
            f.write(f'{newline}')

    def delete_config_file(self):
        file_path = pathlib.Path(self.filename)
        file_path.unlink()


def test_init():
    myFactory = ConfigsTest_Factory()
    myFactory.write_config_file()

    myConfigs = Configs(filename=myFactory.filename)

    assert myConfigs.ofile_base == myFactory.ofile_base

    assert myConfigs.sample_dim == myFactory.sample_dim
    assert myConfigs.xmin == myFactory.xmin
    assert myConfigs.xmax == myFactory.xmax
    assert myConfigs.num_hist_bins == myFactory.num_hist_bins

    assert myConfigs.num_diffusion_steps == myFactory.num_diffusion_steps
    assert myConfigs.dt == myFactory.dt

    assert myConfigs.passive_noise_T == myFactory.passive_noise_T
    assert myConfigs.passive_training_iterations == myFactory.passive_training_iterations

    assert myConfigs.active_noise_Tp == myFactory.active_noise_Tp
    assert myConfigs.active_noise_Ta == myFactory.active_noise_Ta
    assert myConfigs.active_noise_tau == myFactory.active_noise_tau
    assert myConfigs.active_training_iterations == myFactory.active_training_iterations

    assert myConfigs.target_type == myFactory.target_type

    assert myConfigs.mu_list == myFactory.mu_list
    assert myConfigs.sigma_list == myFactory.sigma_list
    assert myConfigs.pi_list == myFactory.pi_list

    myFactory.delete_config_file()
