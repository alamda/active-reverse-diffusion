import configparser


class Configs:
    def __init__(self, filename="diffusion.conf"):
        self.parser = configparser.ConfigParser()
        self.filename = filename

        self.parser.read(filename)

        self.ofile_base = str(self.parser['output']['file_name'])

        self.sample_dim = int(self.parser['sample']['dimension'])
        self.xmin = float(self.parser['sample']['xmin'])
        self.xmax = float(self.parser['sample']['xmax'])
        self.num_hist_bins = int(self.parser['sample']['num_hist_bins'])

        self.num_diffusion_steps = int(self.parser['diffusion']['num_steps'])
        self.dt = float(self.parser['diffusion']['dt'])

        self.passive_noise_T = float(self.parser['passive noise']['T'])
        self.passive_training_iterations = int(
            self.parser['passive noise']['training_iterations'])

        self.active_noise_Tp = float(self.parser['active noise']['Tp'])
        self.active_noise_Ta = float(self.parser['active noise']['Ta'])
        self.active_noise_tau = float(self.parser['active noise']['tau'])
        self.active_training_iterations = int(
            self.parser['active noise']['training_iterations'])

        self.target_type = str(self.parser['target']['type'])

        # Multigaussian target dsn
        if (self.target_type == 'gaussian') or (self.target_type == 'Gaussian'):
            self.target_parser = \
                configparser.ConfigParser(
                    converters={'list': lambda x: [float(i.strip()) for i in x.split(',')]})

            self.target_parser.read(filename)

            self.mu_list = self.target_parser.getlist('target', 'mu_list')
            self.sigma_list = self.target_parser.getlist(
                'target', 'sigma_list')
            self.pi_list = self.target_parser.getlist('target', 'pi_list')
