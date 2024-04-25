import configparser


class Configs:
    def __init__(self, fname="diffusion.conf"):
        self.parser = configparser.ConfigParser()
        self.fname = fname

        self.parser.read(self.fname)

        self.ofile_base = str(self.parser['output']['file_name'])

        # [file_names]        
        self.forward_passive_fname = str(self.parser['file names']['forward_passive'])
        self.forward_active_x_fname = str(self.parser['file names']['forward_active_x'])
        self.forward_active_eta_fname = str(self.parser['file names']['forward_active_eta'])
        self.target_fname = str(self.parser['file names']['target'])
        self.models_passive_fname = str(self.parser['file names']['models_passive'])
        self.models_active_x_fname = str(self.parser['file names']['models_active_x'])
        self.models_active_eta_fname = str(self.parser['file names']['models_active_eta'])

        # [sample]
        self.sample_size = int(self.parser['sample']['size'])
        self.sample_dim = int(self.parser['sample']['dim'])
        self.xmin = float(self.parser['sample']['xmin'])
        self.xmax = float(self.parser['sample']['xmax'])

        if self.sample_dim == 2:
            self.ymin = float(self.parser['sample']['ymin'])
            self.ymax = float(self.parser['sample']['ymax'])

        self.num_hist_bins = int(self.parser['sample']['num_hist_bins'])

        # [diffusion]
        self.diffusion_calculation_type = str(self.parser['diffusion']['calc_type'])
        
        self.diffusion_calculation_type = 'numeric' \
            if self.diffusion_calculation_type is None \
            else self.diffusion_calculation_type
            
        self.num_diffusion_steps = int(self.parser['diffusion']['num_steps'])
        self.dt = float(self.parser['diffusion']['dt'])
        
        # if self.diffusion_calculation_type in ('numeric', 'Numeric'):
        self.passive_training_iters = int(self.parser['diffusion']['passive_training_iters'])
        self.active_training_iters = int(self.parser['diffusion']['active_training_iters'])
        self.reverse_sample_step_interval = int(self.parser['diffusion']['reverse_sample_step_interval'])

        # [passive noise]
        self.passive_noise_T = float(self.parser['passive noise']['T'])
        
        # [active noise]
        self.active_noise_Tp = float(self.parser['active noise']['Tp'])
        self.active_noise_Ta = float(self.parser['active noise']['Ta'])
        self.active_noise_tau = float(self.parser['active noise']['tau'])
       
        # [target]
        self.target_type = str(self.parser['target']['type'])

        # Multigaussian target dsn
        try:
            if self.target_type in('gaussian', 'Gaussian'):
                self.target_parser = \
                    configparser.ConfigParser(
                        converters={'list': lambda x: [float(i.strip()) for i in x.split(',')]})

                self.target_parser.read(self.fname)

                if self.sample_dim == 1:
                    self.mu_list = self.target_parser.getlist('target', 'mu_list')
                elif self.sample_dim == 2:
                    self.mu_x_list = self.target_parser.getlist('target', 'mu_x_list')
                    self.mu_y_list = self.target_parser.getlist('target', 'mu_y_list')
                    
                    if self.mu_y_list is None:
                        self.mu_y_list = [0.0, 0.0]
               
                self.sigma_list = self.target_parser.getlist('target', 'sigma_list')
                self.pi_list = self.target_parser.getlist('target', 'pi_list')

            elif self.target_type in ('quartic', 'double_well'):
                self.a = float(self.parser['target']['a'])
                self.b = float(self.parser['target']['b'])
            else:
                raise ValueError
        except ValueError:
            print("Unknown target type specified in config file, target not configured")
