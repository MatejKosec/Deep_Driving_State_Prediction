#Autonomous driving in TORCS
CS230 Milestone version

Please see model.py:add_prediction_op for the definition of the model. The layout of the model.py is taken from cs224n and adapted.
To run the populate the train set run gym_torcs/populate_replay_buffer.py 
To train the model run train.py (also does evaluate if train is set to false.
The hyperparameters are store in config.py
Past models are graveyarded in retired_models_graveyard.py
The replay buffer is taken from CS234 and ensures each game frame is stored only once (even though 4 frames are needed as input).


