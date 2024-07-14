import jax.numpy as jnp
from .units import UNITS
from .utils import background_dens


# Neutrinos
Tnu_0 = 1.95 * UNITS.K
n_FD_SM = background_dens(p_array)

# Other constants
Grav_const = 4.786486e-20 * UNITS.Mpc / UNITS.MSun
c_light = 299792.458 * UNITS.km / UNITS.s
rho_crit_0_over_hsq = 2.77463e11 * UNITS.MSun / UNITS.Mpc**3
