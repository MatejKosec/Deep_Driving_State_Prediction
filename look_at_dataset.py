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


truth_buffer = train_buffer.sample(6)[0]

plt.figure(1,figsize=(12,8))
subfig = 231
for t in range(6):
    truth = truth_buffer[t,:,:,2]
    print('Truth shape:', truth.shape)
    plt.subplot(231+t)
    plt.imshow(truth.reshape((64,64)),cmap="Greys")
    plt.xlabel('Example: %i'%t)
plt.savefig('./data/sample_inputs4.png',dpi=300, bbox_inches='tight')
    
    