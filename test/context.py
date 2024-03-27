import os
import sys
sys.path.insert(0, os.path.abspath(
    os.path.join(os.path.dirname(__file__), '../src/')))

import data_proc
import diffusion_numeric
import diffusion_analytic
import noise
import read_configs
import target_multi_gaussian
import target_quartic
