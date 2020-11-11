from functools import partial
from jax import lax
from jax import jit, grad, pmap
from jax import random, pmap
import jax.numpy as jnp

@partial(pmap, axis_name='i')
def normalize(x):
  return x / lax.psum(x, 'i')


keys = random.split(random.PRNGKey(0), 8)
mats = pmap(lambda key: random.normal(key, (5000, 6000)))(keys)


print(mats)