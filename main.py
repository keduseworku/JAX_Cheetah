import jax

jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
from jax import jit, vmap
from src.constants import n_FD_SM
import matplotlib.pyplot as plt
from src.ode import *
from src.input import *
from src.distributions import f_today, f_FD
import src.utils as utils
from src.units import UNITS
import time
# import os
# os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"]="false"
# os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"]=".XX"
# os.environ["XLA_PYTHON_CLIENT_ALLOCATOR"]="platform"

@jit
def my_function(u_ini, theta, r_ini, Mh):
    uy_ini = -u_ini * jnp.sin(theta)
    uz_ini = -u_ini * jnp.cos(theta)

    ini_conds = jnp.array([r_ini, 0.0, uy_ini, uz_ini])

    args = (Mh,)

    result = solve_equations(time_span, ini_conds, args, time_eval)
    u_sol = result.ys[-1, 2:]

    return vmap(f_today, in_axes=(None, 0, None))(
        f_FD,
        mass_array,
        jnp.sqrt(jnp.sum(jnp.power(u_sol, 2))),
    )


if __name__ == "__main__":
    print("Solving equations...")
    start = time.time()
    result = vmap(
        vmap(
            vmap(
                vmap(my_function, in_axes=(0, None, None, None)),
                in_axes=(None, 0, None, None),
            ),
            in_axes=(None, None, 0, None),
        ),
        in_axes=(None, None, None, 0),
    )(u_ini_array, theta_array, r_array, Mh_array)
    f_array = result.T
    
    #print(f'f_array shape:{f_array.shape}') 
    print("Solved equations, proceeding to computing overdensities...")

    density_contrast = jnp.zeros((Mh_samples, r_samples, mass_samples))
    z_0 = ((1 + z_end) / 200**(1/3)) - 1
    n_z_factor = (1 + z_0)**3
    n_FD_SM = 3.0 * 1.202 * Tnu_0**3 / 4.0 / jnp.pi**2
    n_FD_SM = n_FD_SM / n_z_factor
    #print(f'the n_z factor:{n_z_factor}')

    for num_Mh in range(Mh_samples):
        for num_r in range(r_samples):
            density_contrast = density_contrast.at[num_Mh, num_r, :].set(
                vmap(utils.density_trapz, in_axes=(0, 0, None))(
                    f_array[:, :, :, num_r, num_Mh],
                    p_array * mass_array[:, None] / mass_fid,
                    theta_array,
                )
                / n_FD_SM
            )
        
            
    end = time.time()
    print("Saving overdensities...")
    jnp.save('overdensity_1e12MSun.npy', density_contrast)
    print("Time elapsed: ", end - start)
    print(density_contrast)

