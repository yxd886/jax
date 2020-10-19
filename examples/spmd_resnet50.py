# Copyright 2018 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""An MNIST example with single-program multiple-data (SPMD) data parallelism.

The aim here is to illustrate how to use JAX's `pmap` to express and execute
SPMD programs for data parallelism along a batch dimension, while also
minimizing dependencies by avoiding the use of higher-level layers and
optimizers libraries.
"""


from functools import partial
import time

import numpy as np
import numpy.random as npr

from jax import jit, grad, pmap
from jax.scipy.special import logsumexp
from jax.lib import xla_bridge
from jax.tree_util import tree_map,tree_flatten,tree_unflatten
from jax import lax
import jax.numpy as jnp
import numpy.random as npr
import jax
import jax.numpy as jnp
from jax import jit, grad, random
from jax.experimental import optimizers
from jax.experimental import stax
from jax.experimental.stax import (AvgPool, BatchNorm, Conv, Dense, FanInSum,
                                   FanOut, Flatten, GeneralConv, Identity,
                                   MaxPool, Relu, LogSoftmax)

def ConvBlock(kernel_size, filters, strides=(2, 2)):
  ks = kernel_size
  filters1, filters2, filters3 = filters
  Main = stax.serial(
      Conv(filters1, (1, 1), strides), BatchNorm(), Relu,
      Conv(filters2, (ks, ks), padding='SAME'), BatchNorm(), Relu,
      Conv(filters3, (1, 1)), BatchNorm())
  Shortcut = stax.serial(Conv(filters3, (1, 1), strides), BatchNorm())
  return stax.serial(FanOut(2), stax.parallel(Main, Shortcut), FanInSum, Relu)


def IdentityBlock(kernel_size, filters):
  ks = kernel_size
  filters1, filters2 = filters
  def make_main(input_shape):
    # the number of output channels depends on the number of input channels
    return stax.serial(
        Conv(filters1, (1, 1)), BatchNorm(), Relu,
        Conv(filters2, (ks, ks), padding='SAME'), BatchNorm(), Relu,
        Conv(input_shape[3], (1, 1)), BatchNorm())
  Main = stax.shape_dependent(make_main)
  return stax.serial(FanOut(2), stax.parallel(Main, Identity), FanInSum, Relu)


# ResNet architectures compose layers and ResNet blocks

def ResNet50(num_classes):
  return stax.serial(
      GeneralConv(('HWCN', 'OIHW', 'NHWC'), 64, (7, 7), (2, 2), 'SAME'),
      BatchNorm(), Relu, MaxPool((3, 3), strides=(2, 2)),
      ConvBlock(3, [64, 64, 256], strides=(1, 1)),
      IdentityBlock(3, [64, 64]),
      IdentityBlock(3, [64, 64]),
      ConvBlock(3, [128, 128, 512]),
      IdentityBlock(3, [128, 128]),
      IdentityBlock(3, [128, 128]),
      IdentityBlock(3, [128, 128]),
      ConvBlock(3, [256, 256, 1024]),
      IdentityBlock(3, [256, 256]),
      IdentityBlock(3, [256, 256]),
      IdentityBlock(3, [256, 256]),
      IdentityBlock(3, [256, 256]),
      IdentityBlock(3, [256, 256]),
      ConvBlock(3, [512, 512, 2048]),
      IdentityBlock(3, [512, 512]),
      IdentityBlock(3, [512, 512]),
      AvgPool((7, 7)), Flatten, Dense(num_classes), LogSoftmax)

if __name__ == "__main__":
  rng_key = random.PRNGKey(0)

  batch_size = 64*4
  num_classes = 1001
  input_shape = (224, 224, 3, batch_size)
  step_size = 0.1
  num_steps = 10

  init_fun, predict_fun = ResNet50(num_classes)
  _, init_params = init_fun(rng_key, input_shape)

  num_devices = xla_bridge.device_count()


  def loss(params, batch):
    inputs, targets = batch
    logits = predict_fun(params, inputs)
    return -jnp.sum(logits * targets)

  def accuracy(params, batch):
    inputs, targets = batch
    target_class = jnp.argmax(targets, axis=-1)
    predicted_class = jnp.argmax(predict_fun(params, inputs), axis=-1)
    return jnp.mean(predicted_class == target_class)

  def synth_batches():
    rng = npr.RandomState(0)
    while True:
      images = rng.rand(*input_shape).astype('float32')
      labels = rng.randint(num_classes, size=(batch_size, 1))
      onehot_labels = labels == jnp.arange(num_classes)

      batch_size_per_device, ragged = divmod(images.shape[-1], num_devices)
      if ragged:
          msg = "batch size must be divisible by device count, got {} and {}."
          raise ValueError(msg.format(batch_size, num_devices))
      shape_prefix = (num_devices, )
      shape_postfix = (batch_size_per_device,)
      images = images.reshape(shape_prefix + images.shape[:-1]+shape_postfix)
      labels = labels.reshape(shape_prefix + shape_postfix+labels.shape[-1:])
      
      yield images, labels


  opt_init, opt_update, get_params = optimizers.momentum(step_size, mass=0.9)
  batches = synth_batches()

  @jit
  def update(i, opt_state, batch):
    params = get_params(opt_state)
    return opt_update(i, grad(loss)(params, batch), opt_state)

  @partial(pmap, axis_name='batch')
  def allreduce_spmd_update( i,op_state, batch):

    #params = tree_unflatten(treedef, params)
    params = get_params(op_state)
    grads = grad(loss)(params, batch)
    leaves, local_treedef = tree_flatten(grads)
    # We compute the total gradients, summing across the device-mapped axis,
    # using the `lax.psum` SPMD primitive, which does a fast all-reduce-sum.
    grads = [lax.psum(dw, 'batch') for dw in leaves]
    grads = tree_unflatten(local_treedef, grads)
    op_state = opt_update(i, grads, op_state)

    return op_state

  @partial(pmap, axis_name='batch')
  def ps_spmd_update( params, batch):
    grads = grad(loss)(params, batch)
    return grads


  @partial(jit, device=jax.devices()[0])
  def ps_pre_process(op_state):
    params = get_params(op_state)
    replicated_op_params = tree_map(replicate_array, params)
    return replicated_op_params

  @partial(jit, device=jax.devices()[0])
  def ps_post_process(grads,op_state,i):
    grads = tree_map(lambda x: jnp.sum(x,axis=0), grads)
    op_state = opt_update(i, grads, op_state)
    return op_state



  replicate_array = lambda x: jnp.broadcast_to(x, (num_devices,) + x.shape)
  allreduce = False

  if allreduce:
    op_state = opt_init(init_params)
    replicated_op_state = tree_map(replicate_array, op_state)
    for i in range(num_steps):
      #params, treedef = tree_flatten(params)
      new_batch = next(batches)
      start_time = time.time()
      replicated_op_state = allreduce_spmd_update( jnp.array([i]*num_devices),replicated_op_state, new_batch)
      end_time = time.time() - start_time
      print("time:",end_time)
  else:
    op_state = opt_init(init_params)
    for i in range (num_steps):
      new_batch = next(batches)
      start_time = time.time()
      replicated_op_params = ps_pre_process(op_state)
      grads = ps_spmd_update(replicated_op_params,new_batch)
      op_state = ps_post_process(grads,op_state,i)
      end_time = time.time() - start_time
      print("time:",end_time)










