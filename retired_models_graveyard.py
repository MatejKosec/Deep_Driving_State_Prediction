"""Implements the core of the model that transforms a batch of input data into predictions. """

#Use convolution to encode the imput image
x1 = tf.contrib.layers.conv2d(self.input_frame_placeholder, 4, \
                                  kernel_size=[4,4], stride=(1,1), padding='SAME', normalizer_fn=tf.contrib.layers.batch_norm)
print('x1 shape', x1.shape)
#x2 = tf.contrib.layers.max_pool2d(x1, \
#                                  kernel_size=[2,2], stride=(2,2), padding='SAME')
x2 = x1
print('x2 shape', x2.shape)
x3 = tf.contrib.layers.conv2d(x2, 4, \
                              kernel_size=[4,4], stride=(1,1), padding='SAME', normalizer_fn=tf.contrib.layers.batch_norm)
print('x3 shape', x3.shape)
filter1 = tf.get_variable('filter', shape=[1,1,4,1],dtype=tf.float32)
x4 = tf.nn.conv2d(x3, filter=filter1, strides=[1,1,1,1], padding='SAME')
print('x4 shape', x4.shape)
#Now combine the encoding the the actoin vector
x5 =tf.contrib.layers.flatten(x4)
print('x5 shape', x5.shape)
x6 =self.input_action_placeholder
print('x6 shape', x6.shape)
x7 =tf.concat([x5,x6],axis=1)
print('x7 shape', x7.shape)

#Predict the future based on image encoding and action taken
pred = tf.contrib.layers.fully_connected(x7, 4096, activation_fn=tf.nn.sigmoid)
print('pred shape', pred.shape)
