import jax
import jax.numpy as jnp
from jax import jit
from .constants import Grav_const
from .input import rho_crit_0, Omega_m, Omega_b, z_end
from .units import UNITS

@jit
def I_func(c):
    return jnp.log(1 + c) - (c / (1 + c))

@jit
def mod_func(redshift):
    ## modulation function for the DM halo growth rate ##
    z_i = z_end
    #200 times critical density at previous redshift
    z_0 = ((1 + z_i) / 200**(1/3)) - 1
    mod = jnp.select([redshift >= z_0], [((z_i - redshift) / (z_i - z_0))], default=1)
    return mod


@jit
def dPhidxi_NFW(pos, redshift, Mh):
    x, y = pos
    r = jnp.sqrt(jnp.power(x, 2.0) + jnp.power(y, 2.0))
    Delta_vir = 200.0
    conc = jax.lax.select(
        redshift >= 4,
        10
        ** (
            1.3081
            - 0.1078 * (1 + redshift)
            + 0.00398 * (1 + redshift) ** 2
            + (0.0223 - 0.0944 * (1 + redshift) ** (-0.3907))
            * jnp.log10(Mh / UNITS.MSun)
        ),
        10
        ** (
            1.7543
            - 0.2766 * (1 + redshift)
            + 0.02039 * (1 + redshift) ** 2
            + (0.2753 + 0.00351 * (1 + redshift) - 0.3038 * (1 + redshift) ** 0.0269)
            * jnp.log10(Mh / UNITS.MSun)
            * (
                1.0
                + (-0.01537 + 0.02102 * (1 + redshift) ** (-0.1475))
                * (jnp.log10(Mh / UNITS.MSun)) ** 2
            )
        ),
    )

    R200 = (3.0 * Mh / (4 * jnp.pi * Delta_vir * rho_crit_0)) ** (1.0 / 3)
    Rs = R200 / conc

    result = jnp.select([r < (R200 * (1 + redshift)), r < (R200 * (1 + z_end))], [(I_func((r / Rs) / (1 + redshift)) / I_func(conc)) - (r / ((1 + z_end) * R200)) ** 3, 1 - (r / ((1 + z_end) * R200)) ** 3], default=0)

    return (
        Grav_const
        * (1 + redshift)
        * (r ** -2)
        * mod_func(redshift)
        * Mh
        * result) * jnp.array([x / r, y / r], dtype=jnp.float64)

@jit
def dPhidxi_hernquist(pos, redshift, Mh):
    y, z = pos
    Mb = (Omega_b / Omega_m) * Mh * mod_func(redshift)
    r = jnp.sqrt(jnp.power(y, 2.0) + jnp.power(z, 2.0))
    Delta_vir = 200.0
    Rvir = (3.0 * Mh / (4 * jnp.pi * Delta_vir * rho_crit_0)) ** (1.0 / 3)
    a = Rvir * 2e-1
    return (Grav_const * Mb / jnp.power(r + a, 2.0)) * jnp.array([y / r, z / r])


@jit
def dPhidxi_tot(pos, redshift, Mh):
    return dPhidxi_NFW(pos, redshift, Mh) #+ dPhidxi_hernquist(pos, redshift, Mh)
