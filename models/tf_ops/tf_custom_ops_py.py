import tensorflow as tf
_reverb_gen_op = tf.load_op_library(
    tf.compat.v1.resource_loader.get_path_to_datafile(
       "libtf_custom_ops_py_gen_op.so"))
_locals = locals()
for k in dir(_reverb_gen_op):
  _locals[k] = getattr(_reverb_gen_op, k)
del _locals
