from abc import ABC as AbstractBaseClass
from abc import abstractmethod


class DiffusionAbstract(AbstractBaseClass):
    def __init__(self, name="diff", total_time=None, dt=None, num_steps=None,
                 noise_list=None, target=None, dim=None):

        self.name = name
        self.set_time_vars(total_time=total_time, dt=dt, num_steps=num_steps)

        self.target = target

        self.noise_list = noise_list
        self.score_fn_list = []
        self.dim = dim

        self.target_sample = None
        self.gen_sample = None

        self.data = None

    def set_time_vars(self, total_time=None, dt=None, num_steps=None):
        try:
            if dt is not None:
                self.dt = dt
            else:
                raise TypeError
        except TypeError:
            print("dt cannot be None")

        try:
            if num_steps is not None:
                self.num_steps = num_steps
                self.total_time = num_steps*self.dt
            elif total_time is not None:
                self.total_time = total_time
                self.num_steps = int(total_time//self.dt) + 1
            else:
                raise TypeError
        except TypeError:
            print("total_time and num_steps cannot both be None")

        try:
            if (self.num_steps is None) or (self.total_time is None) or (self.dt is None):
                raise TypeError
        except TypeError:
            print("One of the necessary time quantities is None")

    def add_target(self, target_obj):
        """Set target data"""
        self.target = target_obj

    def add_noise(self, noise_obj):
        """Define noise objects for the diffusion process"""
        self.noise_list.append(noise_obj)

    def forward_diffuse(self):
        """Forward diffusion process with all noise"""

        try:
            for i in range(self.num_steps):
                if len(self.noise_list > 0):
                    for noise_obj in self.noise_list:
                        if noise_obj.correlation_time is None:
                            contribution = noise_obj.get_diffusion_contribution()
                        else:
                            contribution = self.dt * noise_obj.get_diffusion_contribution()
                            noise_obj.evolve(self.dt)

                        self.data = self.data - self.data*self.dt + contribution
                else:
                    raise AssertionError
        except AssertionError:
            print("Must have at least one type of noise in noise_list")

    def reverse_diffuse(self):
        """Uses score function"""
        try:
            if len(self.score_fn_list) < len(self.noise_list):
                raise AssertionError
        except AssertionError:
            print("Must have score function for each noise to run reverse diffusion")

        data_trj = []

        for time_idx in range(self.num_steps, 0, -1):
            data_trj.append([time_idx, self.data])

            time = time_idx*self.dt

            for (score_fn, noise_obj) in zip(self.score_fn_list, self.noise_list):
                score = score_fn(time=time, noise_obj=noise_obj)

                noise_contribution = noise_obj.get_diffusion_contribution(
                    dt=self.dt)

                self.data += noise_contribution

                noise_obj.update(dt=self.dt)

            self.data += self.dt*self.data

        return data_trj

    # @abstractmethod
    # def sample_reverse(self):
    #     """Generate target data from result of reverse diffusion"""

    # @abstractmethod
    # def compare_to_target(self):
    #     """Compare the sample from reverse diffusion to sample from target dsn"""
