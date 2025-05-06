# @Time     : Jan. 12, 2019 17:45
# @Author   : Veritas YIN
# @FileName : layers.py (Fixed Multi-Scale GLU and Dynamic Dilation with Cropping for AddN)
# @Version  : 2.4
# @IDE      : PyCharm

import tensorflow as tf

def gconv(x, theta, Ks, c_in, c_out):
    kernel = tf.get_collection('graph_kernel')[0]
    n = tf.shape(kernel)[0]
    x_tmp = tf.reshape(tf.transpose(x, [0, 2, 1]), [-1, n])
    x_mul = tf.reshape(tf.matmul(x_tmp, kernel), [-1, c_in, Ks, n])
    x_ker = tf.reshape(tf.transpose(x_mul, [0, 3, 1, 2]), [-1, c_in * Ks])
    x_gconv = tf.reshape(tf.matmul(x_ker, theta), [-1, n, c_out])
    return x_gconv

def layer_norm(x, scope):
    _, _, N, C = x.get_shape().as_list()
    mu, sigma = tf.nn.moments(x, axes=[2, 3], keepdims=True)
    with tf.variable_scope(scope, reuse=tf.compat.v1.AUTO_REUSE):
        gamma = tf.get_variable('gamma', initializer=tf.ones([1, 1, N, C]))
        beta = tf.get_variable('beta', initializer=tf.zeros([1, 1, N, C]))
        _x = (x - mu) / tf.sqrt(sigma + 1e-6) * gamma + beta
    return _x

def temporal_conv_layer(x, Kt, c_in, c_out, act_func='relu', dilation_factors=[1, 2, 4]):
    _, T, n, _ = x.get_shape().as_list()

    if c_in > c_out:
        w_input = tf.get_variable('wt_input', shape=[1, 1, c_in, c_out], dtype=tf.float32)
        tf.add_to_collection('weight_decay', tf.nn.l2_loss(w_input))
        x_input = tf.nn.conv2d(x, w_input, strides=[1, 1, 1, 1], padding='SAME')
    elif c_in < c_out:
        x_input = tf.concat([x, tf.zeros([tf.shape(x)[0], tf.shape(x)[1], tf.shape(x)[2], c_out - c_in])], axis=3)
    else:
        x_input = x

    multi_scale_outputs = []

    for dilation in dilation_factors:
        needed_length = (Kt - 1) * dilation + 1
        input_length = tf.shape(x)[1]
        can_apply = tf.greater_equal(input_length, needed_length)

        def apply_conv():
            with tf.variable_scope(f'dilation_{dilation}'):
                cropped_input = x_input[:, (Kt-1)*dilation:, :, :]
                if act_func == 'GLU':
                    wt = tf.get_variable(f'wt_d{dilation}', shape=[Kt, 1, c_in, 2*c_out], dtype=tf.float32)
                    bt = tf.get_variable(f'bt_d{dilation}', initializer=lambda: tf.zeros([2*c_out]), dtype=tf.float32)
                    x_conv = tf.nn.convolution(x, wt, padding='VALID', dilation_rate=[dilation, 1]) + bt
                    P = x_conv[:, :, :, :c_out]
                    Q = x_conv[:, :, :, c_out:]
                    out = (P + cropped_input) * tf.nn.sigmoid(Q)
                else:
                    wt = tf.get_variable(f'wt_d{dilation}', shape=[Kt, 1, c_in, c_out], dtype=tf.float32)
                    bt = tf.get_variable(f'bt_d{dilation}', initializer=lambda: tf.zeros([c_out]), dtype=tf.float32)
                    x_conv = tf.nn.convolution(x, wt, padding='VALID', dilation_rate=[dilation, 1]) + bt
                    if act_func == 'linear':
                        out = x_conv
                    elif act_func == 'sigmoid':
                        out = tf.nn.sigmoid(x_conv)
                    elif act_func == 'relu':
                        out = tf.nn.relu(x_conv + cropped_input)
                    else:
                        raise ValueError(f'ERROR: activation function "{act_func}" is not defined.')
                return out

        out = tf.cond(can_apply, apply_conv, lambda: tf.zeros_like(x_input[:, 0:1, :, :]))
        multi_scale_outputs.append(out)

    min_time = tf.reduce_min([tf.shape(o)[1] for o in multi_scale_outputs])
    multi_scale_outputs = [o[:, :min_time, :, :] for o in multi_scale_outputs]

    x_out = tf.add_n(multi_scale_outputs) / len(multi_scale_outputs)
    return x_out

def spatio_conv_layer(x, Ks, c_in, c_out):
    n = tf.shape(x)[2]
    T_real = tf.shape(x)[1]

    if c_in > c_out:
        w_input = tf.get_variable('ws_input', shape=[1,1,c_in,c_out], dtype=tf.float32)
        tf.add_to_collection('weight_decay', tf.nn.l2_loss(w_input))
        x_input = tf.nn.conv2d(x, w_input, strides=[1,1,1,1], padding='SAME')
    elif c_in < c_out:
        x_input = tf.concat([x, tf.zeros([tf.shape(x)[0], T_real, n, c_out-c_in])], axis=3)
    else:
        x_input = x

    ws = tf.get_variable('ws', shape=[Ks*c_in, c_out], dtype=tf.float32)
    bs = tf.get_variable('bs', initializer=tf.zeros([c_out]), dtype=tf.float32)

    tf.add_to_collection('weight_decay', tf.nn.l2_loss(ws))
    variable_summaries(ws, 'theta')

    x_gconv = gconv(tf.reshape(x, [-1, n, c_in]), ws, Ks, c_in, c_out) + bs
    x_gc = tf.reshape(x_gconv, [-1, T_real, n, c_out])

    return tf.nn.relu(x_gc + x_input)

def st_conv_block(x, Ks, Kt, channels, scope, keep_prob, act_func='GLU', dilation_factors=[1,2,4]):
    c_si, c_t, c_oo = channels
    with tf.variable_scope(f'stn_block_{scope}_in', reuse=tf.compat.v1.AUTO_REUSE):
        x_s = temporal_conv_layer(x, Kt, c_si, c_t, act_func=act_func, dilation_factors=dilation_factors)
        x_t = spatio_conv_layer(x_s, Ks, c_t, c_t)
    with tf.variable_scope(f'stn_block_{scope}_out', reuse=tf.compat.v1.AUTO_REUSE):
        x_o = temporal_conv_layer(x_t, Kt, c_t, c_oo, act_func=act_func, dilation_factors=dilation_factors)
    x_ln = layer_norm(x_o, f'layer_norm_{scope}')
    return tf.nn.dropout(x_ln, keep_prob)

def fully_con_layer(x, n, channel, scope):
    with tf.variable_scope(scope, reuse=tf.compat.v1.AUTO_REUSE):
        w = tf.get_variable(name=f'w_{scope}', shape=[1,1,channel,1], dtype=tf.float32)
        b = tf.get_variable(name=f'b_{scope}', initializer=tf.zeros([n,1]), dtype=tf.float32)
        tf.add_to_collection('weight_decay', tf.nn.l2_loss(w))
        return tf.nn.conv2d(x, w, strides=[1,1,1,1], padding='SAME') + b

def output_layer(x, T, scope, act_func='GLU'):
    _, _, n, channel = x.get_shape().as_list()
    with tf.variable_scope(f'{scope}_in', reuse=tf.compat.v1.AUTO_REUSE):
        x_i = temporal_conv_layer(x, T, channel, channel, act_func=act_func)
    x_ln = layer_norm(x_i, f'layer_norm_{scope}')
    with tf.variable_scope(f'{scope}_out', reuse=tf.compat.v1.AUTO_REUSE):
        x_o = temporal_conv_layer(x_ln, 1, channel, channel, act_func='sigmoid')
    x_fc = fully_con_layer(x_o, n, channel, scope)
    return x_fc

def variable_summaries(var, v_name):
    with tf.name_scope('summaries'):
        mean = tf.reduce_mean(var)
        tf.summary.scalar(f'mean_{v_name}', mean)
        with tf.name_scope(f'stddev_{v_name}'):
            stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
        tf.summary.scalar(f'stddev_{v_name}', stddev)
        tf.summary.scalar(f'max_{v_name}', tf.reduce_max(var))
        tf.summary.scalar(f'min_{v_name}', tf.reduce_min(var))
        tf.summary.histogram(f'histogram_{v_name}', var)
