import tensorflow as tf
import numpy as np

#Model skeleton taken from CS224N    
class Model(object):
    def __init__(self,config):
        self.build(config)
    
    def add_placeholders(self):
        
        self.output_frame_placeholder = tf.placeholder(dtype=tf.float32, shape=[None,64,64])
        self.input_frame_placeholder = tf.placeholder(dtype=tf.float32, shape=[None,64,64,3])
        self.input_action_placeholder = tf.placeholder(dtype=tf.float32, shape=[None,1])
        

    def create_feed_dict(self, obs_batch, actions_batch=None):
        """Creates the feed_dict for one step of training. """
        feed_dict = {self.output_frame_placeholder: obs_batch[:,:,:,-1], #future 
                     self.input_frame_placeholder: obs_batch[:,:,:,0:-1], #past
                     self.input_action_placeholder: actions_batch} #action set includes future action
        
        return feed_dict

    def add_prediction_op(self):
        """Implements the core of the model that transforms a batch of input data into predictions. """
        
        #Convolution on the input data 
            #(cnn,batch norm, relu) 
            
            #(cnn,batch norm, relu) 
            
            #maxpool
        x1 =tf.contrib.layers.flatten(self.input_frame_placeholder)
        print('x1 shape', x1.shape)
        x2 =self.input_action_placeholder
        print('x2 shape', x2.shape)
        x3 =tf.concat([x1,x2],axis=1)
        print('x3 shape', x3.shape)
        x4 = tf.contrib.layers.fully_connected(x3, 8000)
        pred = tf.contrib.layers.fully_connected(x4, 4096)
        
        
        
        #Fully connected hidden layer on the actions
        
        #Merge the two streams
        
        #Unconvolution net
        
        return pred
        

    def add_loss_op(self, pred):
        
        loss = tf.reduce_mean((pred-tf.contrib.layers.flatten(self.output_frame_placeholder))**2)
        
        return loss

    def add_training_op(self, loss):
  
        return tf.train.AdamOptimizer(self.config.lr).minimize(loss)
        

    def train_on_batch(self, sess, observations_batch, actions_batch):
        """Perform one step of gradient descent on the provided batch of data. """
        feed = self.create_feed_dict(observations_batch, actions_batch=actions_batch)
        _, loss = sess.run([self.train_op, self.loss], feed_dict=feed)
        return loss

    def loss_on_batch(self, sess, observations_batch, actions_batch):
        """Make predictions for the provided batch of data """
        feed = self.create_feed_dict(observations_batch, actions_batch=np.reshape(actions_batch,[actions_batch.shape[0],1]))
        loss = sess.run([self.loss], feed_dict=feed)
        return loss

    def build(self, config):
        self.config = config
        self.add_placeholders()
        self.pred = self.add_prediction_op()
        self.loss = self.add_loss_op(self.pred)
        self.train_op = self.add_training_op(self.loss)
        
        
    def run_epoch(self, sess, train_buffer, dev_buffer):
        n_minibatches = train_buffer.num_in_buffer // self.config.batch_size
        prog = tf.keras.utils.Progbar(target=n_minibatches)
        for i in range(n_minibatches):
            obs_batch, act_batch, rew_batch, next_obs_batch, done_mask = train_buffer.sample(self.config.batch_size)
            loss = self.train_on_batch(sess, obs_batch, act_batch)
            prog.update(i + 1, [("train loss", loss)], force=i + 1 == n_minibatches)

        print("Evaluating on dev set",)
        obs_batch, act_batch, rew_batch, next_obs_batch, done_mask = dev_buffer.sample(dev_buffer.num_in_buffer-1)
        dev_loss, _ = self.loss_on_batch(sess, obs_batch,act_batch)
        print("Dev score: {:.2f}".format(dev_loss))
        return dev_loss

    def fit(self, sess, saver, train_buffer, dev_buffer):
        best_dev_loss = 0
        for epoch in range(self.config.n_epochs):
            print("Epoch {:} out of {:}".format(epoch + 1, self.config.n_epochs))
            dev_loss = self.run_epoch(sess, train_buffer, dev_buffer)
            if dev_loss < best_dev_loss:
                best_dev_loss = dev_loss
                if saver:
                    print("New best dev loss! Saving model in ./data/weights/parser.weights")
                    saver.save(sess, './data/weights/parser.weights')
            print()
