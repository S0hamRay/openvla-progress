# Sparse Autoencoders in SmolVLA

Kinda self-explanatory title, but there aren't any sparse autoencoders at the time of writing this. This uses a pybullet simulation environment and SmolVLA for control. SmolVLA pretraining
was done as mentioned in the docs, transfer learning was done with 50 episodes (might train some more eventually). Python 3.11 works well, only other installations are pybullet, lerobot dependencies 
(also mentioned in the docs) and some basic stuff for language input to the model.

`record_dataset.py` and `converter.py` are for transfer learning, they can be safely ignored

Stuff to think about:

1) The robot has more joints than SmolVLA expects so some are not being controlled (shouldn't be a major issue hopefully)
