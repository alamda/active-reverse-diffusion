from diffusion_numeric import DiffusionNumeric
from noise_passive import NoisePassive
from noise_active import NoiseActive
from target_multi_gaussian import TargetMultiGaussian
from data_proc import DataProc

import pickle

if __name__ == "__main__":
    sample_dim = 1000
    ofile_base = "abc"
    num_diffusion_steps = 100
    dt = 0.01

    T_passive = 0.5
    T_active = 0.5
    tau = 0.1

    mu_list = [-2.0, 0.0, 2.0]
    sigma_list = [0.5, 0.5, 0.5]
    pi_list = [1.0, 1.0, 1.0]

    xmin = -5
    xmax = 5

    myPassiveNoise = NoisePassive(T=T_passive,
                                  dim=sample_dim)

    myActiveNoise = NoiseActive(T=T_active,
                                tau=tau,
                                dim=sample_dim)

    myTarget = TargetMultiGaussian(mu_list=mu_list,
                                   sigma_list=sigma_list,
                                   pi_list=pi_list,
                                   dim=sample_dim)

    myDataProc = DataProc(xmin=xmin, xmax=xmax)

    myDiffNum = DiffusionNumeric(ofile_base=ofile_base,
                                 passive_noise=myPassiveNoise,
                                 active_noise=myActiveNoise,
                                 target=myTarget,
                                 num_diffusion_steps=num_diffusion_steps,
                                 dt=dt,
                                 sample_dim=sample_dim,
                                 data_proc=myDataProc)

    myDiffNum.train_diffusion_passive()
    myDiffNum.sample_from_diffusion_passive()
    myDiffNum.calculate_passive_diff_list()

    myDiffNum.train_diffusion_active()
    myDiffNum.sample_from_diffusion_active()
    myDiffNum.calculate_active_diff_list()

    with open(f"{ofile_base}.pkl", 'wb') as f:
        pickle.dump(myDiffNum, f)

    breakpoint()
