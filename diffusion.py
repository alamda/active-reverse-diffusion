import abc.ABC


class DiffusionAbstract(abc.ABC):
    def __init__(self, name="diff"):
        self.name = name

        self.target = None
        self.noise_list = []

        self.score_fn = None

        self.target_sample = None
        self.gen_sample = None

    def add_target(self, target_obj):
        """Set target data"""
        self.target = target_obj

    def add_noise(self, noise_obj):
        """Define noise objects for the diffusion process"""
        self.noise_list.append(noise_obj)

    def forward_diffuse(self, data, dt):
        """Forward diffusion process with all noise"""

        try:
            if len(self.noise_list > 0):
                for noise_obj in self.noise_list:
                    if noise_obj.correlation_time is None:
                        contribution = noise_obj.get_diffusion_contribution()
                    else:
                        contribution = dt * noise_obj.get_diffusion_contribution()
                        noise_obj.evolve()

                    data = data - data*dt + contribution
            else:
                raise AssertionError
        except AssertionError:
            print("Must have at least one type of noise in noise_list")

    @abstractmethod
    def reverse_diffuse(self):
        """Uses score function"""

    @abstractmethod
    def sample_reverse(self):
        """Generate target data from result of reverse diffusion"""

    @abstractmethod
    def compare_to_target(self):
        """Compare the sample from reverse diffusion to sample from target dsn"""
