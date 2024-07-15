import jax
import jax.numpy as jnp
from jax import jit
from .constants import Grav_const
from .input import rho_crit_0, z_end, Omega_m, Ri, r200, rs
from .units import UNITS

@jit
def I_func(c):
    return jnp.log(1 + c) - (c / (1 + c))


@partial(jit, static_argnums=(1,))
def mod_func(redshift, power=1):
    ## modulation function for the DM halo growth rate. Power is optional arg, 1 for linear growth by default ##
    z_i = z_end
    #200 times critical density at previous redshift
    z_0 = ((1 + z_i) / 200**(1/3)) - 1
    mod = jnp.select([redshift <= z_0, redshift < z_i], [1, ((z_i - redshift) / (z_i - z_0))**power], default=0)
    return mod


@jit
def dPhidxi_NFW(pos, redshift, Mh):
    y, z = pos
    r = jnp.sqrt(jnp.power(y, 2.0) + jnp.power(z, 2.0))
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

    #R200 = (3.0 * Mh / (4 * jnp.pi * Delta_vir * rho_crit_0 * Omega_m)) ** (1.0 / 3)
    rs = r200 / conc

    result = jnp.select([r < (r200 * (1 + redshift)), r < (r200 * (1 + z_end))], [(I_func((r / rs) / (1 + redshift)) / I_func(conc)) - (r / ((1 + z_end) * r200)) ** 3, 1 - (r / ((1 + z_end) * r200)) ** 3], default=0)

    return (
        Grav_const
        * (1 + redshift)
        * (r ** -2)
        * mod_func(redshift)
        * Mh
        * result) * jnp.array([y / r, z / r], dtype=jnp.float64)


@jit
def dPhidxi_tot(pos, redshift, Mh):
    return dPhidxi_NFW(pos, redshift, Mh)
