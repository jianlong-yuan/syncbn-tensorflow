# syncbn-tensorflow

Syncronized Batch Normalization

syncbn-tensorflow easy use


''' python

@add_arg_scope
def sync_batch_norm(inputs,
                    decay=0.999,
                    center=True,
                    scale=False,
                    epsilon=0.001,
                    activation_fn=None,
                    param_initializers=None,
                    param_regularizers=None,
                    updates_collections=ops.GraphKeys.UPDATE_OPS,
                    is_training=True,
                    reuse=None,
                    variables_collections=None,
                    outputs_collections=None,
                    trainable=True,
                    batch_weights=None,
                    fused=None,
                    data_format=DATA_FORMAT_NHWC,
                    zero_debias_moving_mean=False,
                    scope=None,
                    renorm=False,
                    renorm_clipping=None,
                    renorm_decay=0.99,
                    adjustment=None,
                    num_dev=1):
// num_dev is how many gpus you use.

  import tensorflow as tf
  from tensorflow.contrib.nccl.ops import gen_nccl_ops
  from tensorflow.contrib.framework import add_model_variable
  import re

  if data_format not in ['NHWC']:
    msg_str = "Only support NHWC and NCHW format. %s is an unknown data format." % data_format
    raise TypeError(msg_str)

  red_axises = [0, 1, 2]
  num_outputs = inputs.get_shape().as_list()[-1]

  if scope is None:
    scope = 'BatchNorm'

  layer_variable_getter = _build_variable_getter()
  with variable_scope.variable_scope(
      scope,
      'BatchNorm',
      reuse=reuse,
      custom_getter=layer_variable_getter) as sc:

    gamma = tf.get_variable(name='gamma', shape=[num_outputs], dtype=tf.float32,
                            initializer=tf.constant_initializer(1.0), trainable=trainable,
                            collections=variables_collections)

    beta  = tf.get_variable(name='beta', shape=[num_outputs], dtype=tf.float32,
                            initializer=tf.constant_initializer(0.0), trainable=trainable,
                            collections=variables_collections)

    moving_mean = tf.get_variable(name='moving_mean', shape=[num_outputs], dtype=tf.float32,
                                initializer=tf.constant_initializer(0.0), trainable=False,
                                collections=variables_collections)
                                
    moving_var = tf.get_variable(name='moving_variance', shape=[num_outputs], dtype=tf.float32,
                                initializer=tf.constant_initializer(1.0), trainable=False,
                                collections=variables_collections)

    global _Counter_syncbn

    if is_training and trainable:
      
      if num_dev == 1:
        mean, var = tf.nn.moments(inputs, red_axises)
      else:
        shared_name = re.sub('tower[0-9]+/', '', tf.get_variable_scope().name)
        if _Counter_syncbn ==0:
          print('using syncbn')
        batch_mean        = tf.reduce_mean(inputs, axis=red_axises)
        batch_mean_square = tf.reduce_mean(tf.square(inputs), axis=red_axises)
        batch_mean        = gen_nccl_ops.nccl_all_reduce(
          input=batch_mean,
          reduction='sum',
          num_devices=num_dev,
          shared_name=shared_name + '_NCCL_mean') * (1.0 / num_dev)
        batch_mean_square = gen_nccl_ops.nccl_all_reduce(
          input=batch_mean_square,
          reduction='sum',
          num_devices=num_dev,
          shared_name=shared_name + '_NCCL_mean_square') * (1.0 / num_dev)
        mean              = batch_mean
        var               = batch_mean_square - tf.square(batch_mean)

      outputs = tf.nn.batch_normalization(inputs, mean, var, beta, gamma, epsilon)

      if int(outputs.device[-1])== 0:
        _Counter_syncbn = _Counter_syncbn +1
        if _Counter_syncbn ==1:
          print('update moving mean and variance :{}'.format(outputs))
        update_moving_mean_op = tf.assign(moving_mean, moving_mean * decay + mean * (1 - decay))
        update_moving_var_op  = tf.assign(moving_var,  moving_var  * decay + var  * (1 - decay))
        add_model_variable(moving_mean)
        add_model_variable(moving_var)
        
        if updates_collections is None:
          with tf.control_dependencies([update_moving_mean_op, update_moving_var_op]):
            outputs = tf.identity(outputs)
        else:
          ops.add_to_collections(updates_collections, update_moving_mean_op)
          ops.add_to_collections(updates_collections, update_moving_var_op)
          outputs = tf.identity(outputs)
      else:
        outputs = tf.identity(outputs)

    else:
      outputs,_,_ = nn.fused_batch_norm(inputs, gamma, beta, mean=moving_mean, variance=moving_var, epsilon=epsilon, is_training=False)

    if activation_fn is not None:
      outputs = activation_fn(outputs)

    return utils.collect_named_outputs(outputs_collections, sc.name, outputs)
    
    '''
