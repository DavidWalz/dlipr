__all__ = [
    "utils",
    "cifar",
    "mnist",
    "svhn",
    "ising",
    "flower",
    "speckles"]

import dlipr.utils
import dlipr.cifar
import dlipr.mnist
import dlipr.svhn
import dlipr.ising
import dlipr.flower
import dlipr.speckles
import tensorflow as tf
default_init = tf.Session.__init__

mem_frac = 0.45

def init_with_gpu_frac(self, *args, **kwargs):
    kwargs['config'] = tf.ConfigProto(gpu_options=tf.GPUOptions(per_process_gpu_memory_fraction=mem_frac))
    default_init(self, *args, **kwargs)


tf.Session.__init__ = init_with_gpu_frac
