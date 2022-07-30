# Change a photo to cartoon style
## Description
This repository give a slolution to change a real image to a cartoon-style image.
## Process data for training
- Put your real-images in _**dataset/trainA**_ dir and cartoon-images in _**dataset/trainB**_ dir. </br>
- Run _**utils/get_smooth_imgs.py**_ to creat smooth-images from cartoon-images (default size of these squared-smooth images is 1024).
## Train
- Run _**train.py**_, or you can run this block:
  ```sh 
  from train import Train
  # Choose your-train size, epochs, steps_per_epoch
  train = Train(size, epochs, steps_per_epoch)
  train.train()
  ```
 - Weights will be saved in _**weights**_ dir.
 ## Test

