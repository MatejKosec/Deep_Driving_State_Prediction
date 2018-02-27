#The file used to train the model
from config import Config
from gym_torcs.get_buffer import GetBuffer
from matplotlib import pyplot as plt

replay_file_train = './data/replay_buffer_train.pkl'
replay_file_dev = './data/replay_buffer_dev.pkl'
replay_file_test = './data/replay_buffer_test.pkl'

#Get the databuffers
train_buffer = GetBuffer(replay_file_train)
dev_buffer   = GetBuffer(replay_file_dev)


truth = train_buffer.sample(1)[0]
truth = truth[0,:,:,-1]
print('Truth shape:', truth.shape)


plt.figure(1,figsize=(4,4))
plt.imshow(truth.reshape((64,64)),cmap="Greys")
plt.xlabel('Truth')