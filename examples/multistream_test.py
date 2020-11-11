from functools import partial
from jax import lax
from jax import jit, grad, pmap
import jax.numpy as jnp

@partial(pmap, axis_name='i')
def normalize(x):
  return x / lax.psum(x, 'i')

print(normalize(jnp.arange(4.)))