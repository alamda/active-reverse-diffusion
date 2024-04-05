import os
import sys

sys.path.insert(0, os.path.abspath(
    os.path.join(os.path.dirname(__file__), '../src/')))

import data_proc
import diffusion
import diffusion_numeric
import noise
