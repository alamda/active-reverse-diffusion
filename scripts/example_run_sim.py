from diffusion_numeric import DiffusionNumeric
from noise import NoiseActive, NoisePassive
from target_multi_gaussian import TargetMultiGaussian
from data_proc import DataProc

import pickle

if __name__ == "__main__":
    ofile_base = "data"

    sample_dim = 80000
    num_diffusion_steps = 160
    dt = 0.005

    passive_noise_T = 1.0

    active_noise_Tp = 0
    active_noise_Ta = 1.0
    tau = 0.25

    mu_list = [-1.2, 1.2]
    sigma_list = [1.0, 1.0]
    pi_list = [1.0, 1.0]

    xmin = -5
    xmax = 5

    myPassiveNoise = NoisePassive(T=passive_noise_T,
                                  dim=sample_dim)

    myActiveNoise = NoiseActive(Tp=active_noise_Tp,
                                Ta=active_noise_Ta,
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
