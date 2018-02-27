import tensorflow as tf
import numpy as np
import functools
import os

#Model skeleton taken from CS224N    
class Model(object):
    def __init__(self,config):
        self.build(config)
        
    def add_writers(self, train_writer, eval_writer):
        self.train_writer = train_writer
        self.eval_writer = eval_writer
        print('Summary writers add to the graph')
    
    def add_placeholders(self):
        
        self.output_frame_placeholder = tf.placeholder(dtype=tf.float32, shape=[None,64,64])
        self.input_frame_placeholder = tf.placeholder(dtype=tf.float32, shape=[None,64,64,3])
        self.input_action_placeholder = tf.placeholder(dtype=tf.float32, shape=[None,1])
        

    def create_feed_dict(self, obs_batch, actions_batch=None):
        """Creates the feed_dict for one step of training. """
        feed_dict = {self.output_frame_placeholder: obs_batch[:,:,:,-1], #future 
                     self.input_frame_placeholder: obs_batch[:,:,:,0:-1], #past
                     self.input_action_placeholder: np.reshape(actions_batch,[actions_batch.shape[0],1])} #action set includes future action
        
        return feed_dict

    def add_prediction_op(self):
        """Implements the core of the model that transforms a batch of input data into predictions. """
        
        #Use convolution to encode the imput image
        x1 = tf.contrib.layers.conv2d(self.input_frame_placeholder, 8, \
                                          kernel_size=[8,8], stride=(1,1), padding='SAME', normalizer_fn=tf.contrib.layers.batch_norm)
        print('x1 shape', x1.shape)
        x2 = tf.contrib.layers.conv2d(x1, 8, \
                                          kernel_size=[4,4], stride=(1,1), padding='SAME', normalizer_fn=tf.contrib.layers.batch_norm)
        print('x2 shape', x2.shape)
        #x2 = tf.contrib.layers.max_pool2d(x1, \
        #                                  kernel_size=[2,2], stride=(2,2), padding='SAME')
        #print('x2 shape', x2.shape)
        x5 = x2
        
        
        #x5 = tf.image.resize_bicubic(x4, [48,48])
        print('x5 shape', x5.shape)
        x6= tf.contrib.layers.conv2d_transpose(x5, 6 , [6,6], stride=1, padding='SAME')
        print('x6 shape', x6.shape)
        x6= tf.contrib.layers.conv2d_transpose(x6, 6 , [4,4], stride=1, padding='SAME')
        print('x6 shape', x6.shape)
        x7 = tf.image.resize_bilinear(x6, [64,64])
        print('x7 shape', x7.shape)
        x8= tf.contrib.layers.conv2d_transpose(x7, 4 ,  [4,4], stride=1, padding='SAME',activation_fn=tf.nn.relu)
        print('x8 shape', x8.shape)
        x8= tf.contrib.layers.conv2d_transpose(x8, 1 ,  [2,2], stride=1, padding='SAME',activation_fn=tf.nn.sigmoid)
        print('x8 shape', x8.shape)
        pred=tf.contrib.layers.flatten(x8)
        print('pred shape', pred.shape)
            

        
        return pred
        

    def add_loss_op(self, pred):
        loss = tf.reduce_mean((pred-tf.contrib.layers.flatten(self.output_frame_placeholder))**2)
        tf.summary.scalar("loss", loss)
        self.summary_op = tf.summary.merge_all()
        return loss

    def add_training_op(self, loss):
        self.global_step = tf.Variable(0,dtype=tf.int32,trainable=False,name='global_step')
        return tf.train.AdamOptimizer(self.config.lr).minimize(loss,global_step=self.global_step)
        

    def train_on_batch(self, sess, observations_batch, actions_batch):
        """Perform one step of gradient descent on the provided batch of data. """
        feed = self.create_feed_dict(observations_batch, actions_batch=actions_batch)
        _, loss, summary, global_step = sess.run([self.train_op, self.loss, self.summary_op,self.global_step], feed_dict=feed)
        self.train_writer.add_summary(summary, global_step=global_step)
        return loss

    def loss_on_batch(self, sess, observations_batch, actions_batch):
        """Make predictions for the provided batch of data """
        feed = self.create_feed_dict(observations_batch, actions_batch=actions_batch)
        loss, summary, global_step = sess.run([self.loss,self.summary_op,self.global_step], feed_dict=feed)
        self.eval_writer.add_summary(summary, global_step=global_step)
        return loss

    def build(self, config):
        self.config = config
        self.add_placeholders()
        self.pred = self.add_prediction_op()
        self.loss = self.add_loss_op(self.pred)
        self.train_op = self.add_training_op(self.loss)
        
    def compare_outputs(self, test_buffer, sess):
        samples = test_buffer.sample(self.config.n_test_samples)
        obs_batch, act_batch, rew_batch, next_obs_batch, done_mask = samples
        feed = self.create_feed_dict(obs_batch, actions_batch=act_batch)
        predictions = sess.run([self.pred], feed_dict=feed)[0]
        from matplotlib import pyplot as plt
        for i in range(self.config.n_test_samples):
            truth = obs_batch[i,:,:,-1]
            print('Truth shape:', truth.shape)
            prediction = predictions[i]
            print('Predictions shape:', predictions.shape)
            print('Prediction shape:', prediction.shape)
            
            plt.figure(1,figsize=(10,5))
            plt.title('Comparison between prediction and truth (test set)')
            plt.subplot(121)
            plt.imshow(prediction.reshape((64,64)),cmap="Greys")
            plt.xlabel('Prediction') #1
            plt.subplot(122)
            plt.imshow(truth.reshape((64,64)),cmap="Greys")
            plt.xlabel('Truth')
            plt.savefig('./data/example_figure%i.png'%i,dpi=300, bbox_inches='tight')
        
    def run_epoch(self, sess, train_buffer, dev_buffer):
        n_minibatches = train_buffer.num_in_buffer // self.config.batch_size
        prog = tf.keras.utils.Progbar(target=n_minibatches)
        for i in range(n_minibatches):
            obs_batch, act_batch, rew_batch, next_obs_batch, done_mask = train_buffer.sample(self.config.batch_size)
            loss = self.train_on_batch(sess, obs_batch, act_batch)
            prog.update(i + 1, [("train loss", loss)], force=i + 1 == n_minibatches)
            if i%100 == 0 or i== n_minibatches -1:
                if i== n_minibatches -1: print("Evaluating on dev set",) 
                obs_batch, act_batch, rew_batch, next_obs_batch, done_mask = dev_buffer.sample(dev_buffer.num_in_buffer-1)
                dev_loss = self.loss_on_batch(sess, obs_batch,act_batch)
                if i==n_minibatches -1: print("Dev loss: {:.7f}".format(dev_loss))
        return dev_loss

    def fit(self, sess, saver,train_buffer, dev_buffer):
        best_dev_loss = 100
        for epoch in range(self.config.n_epochs):
            print("Epoch {:} out of {:}".format(epoch + 1, self.config.n_epochs))
            dev_loss = self.run_epoch(sess, train_buffer, dev_buffer)
            if dev_loss < best_dev_loss:
                best_dev_loss = dev_loss
                if saver:
                    print("New best dev loss! Saving model in ./data/weights/predictor.weights")
                    saver.save(sess, './data/weights/predictor.weights')
            print()
            
            
    def count_trainable_params(self):
        shapes = [functools.reduce(lambda x,y: x*y,variable.get_shape()) for variable in tf.trainable_variables()]
        return functools.reduce(lambda x,y: x+y, shapes)
