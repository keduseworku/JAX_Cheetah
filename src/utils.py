import jax
import jax.numpy as jnp
from jax import jit
import numpy as np
from .constants import c_light
from .units import UNITS
from .input import Omega_m, h, z_end
from functools import partial


points, weights = np.polynomial.legendre.leggauss(25)


@partial(jit, static_argnums=0)
def integrator(f, a, b):
    sub = (b - a) / 2.0
    add = (b + a) / 2.0

    return jax.lax.select(sub == 0, 0.0, sub * jnp.dot(f(sub * points + add), weights))


@jit
def Hubble(z):
    return (100.0 * h * UNITS.km / UNITS.s / UNITS.Mpc) * jnp.sqrt(Omega_m * jnp.power(1 + z, 3) + 1.0 - Omega_m)



@jit
def density_trapz(f_array, p_array, phi_array):
    mu_integral = jax.scipy.integrate.trapezoid(f_array, jnp.cos(phi_array), axis=-1)
    
    #mu_integral is negative because our theta range goes is [1,-1] while mu is [-1,1]
    p_integral = jax.scipy.integrate.trapezoid(-mu_integral * p_array**2, p_array)
    return p_integral / (2.0 * np.pi)**2

