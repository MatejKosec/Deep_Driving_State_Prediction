#The file used to train the model
import tensorflow as tf
from model import Model
from config import Config
import os
import time
from gym_torcs.get_buffer import GetBuffer


debug = False
train = 

print(80 * "=")
print("CS 230 Miterm project report".center(80))
print("MatejKosec".center(80))
print(80 * "=")


#Data locations 
replay_file_train = './data/replay_buffer_train.pkl'
replay_file_dev = './data/replay_buffer_dev.pkl'
replay_file_test = './data/replay_buffer_test.pkl'

#Get the databuffers
train_buffer = GetBuffer(replay_file_train)
dev_buffer   = GetBuffer(replay_file_dev)


#Result locations

config = Config()


if not os.path.exists('./data/weights/'):
    os.makedirs('./data/weights/')

with tf.Graph().as_default() as graph:
    print("Building model..."),
    start = time.time()
    model = Model(config)
    print('Model has {} parameters'.format(model.count_trainable_params()))
    #Setup variable initialization
    init_op = tf.global_variables_initializer()
    saver = None if debug else tf.train.Saver()
    print("took {:.2f} seconds\n".format(time.time() - start))
graph.finalize()

with tf.Session(graph=graph) as session:
    if train:
        train_writer = tf.summary.FileWriter(os.path.join(config.results_dir, 'train_summaries'), session.graph)
        eval_writer = tf.summary.FileWriter(os.path.join(config.results_dir, 'eval_summaries'), session.graph)
        model.add_writers(train_writer, eval_writer)
        
        session.run(init_op)
        print(80 * "=")
        print("TRAINING".center(80))
        print(80 * "=")
        model.fit(session, saver, train_buffer, dev_buffer)

    if not debug:
        print(80 * "=")
        print("TESTING".center(80))
        print(80 * "=")
        
        print("Restoring the best model weights found on the dev set")
        saver.restore(session, './data/weights/predictor.weights')
        print("Final evaluation on test set")
        test_buffer   = GetBuffer(replay_file_test)
        model.compare_outputs(test_buffer, session)
        print("Done! Took {:.2f} seconds for everything \n".format(time.time() - start))


