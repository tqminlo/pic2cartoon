# Change a photo to cartoon style
## Description
This repository give a slolution to change a real image to a cartoon-style image. </br>
Model was trained with 6227 real images and 2582 cartoon images, with 10 epochs. This project is still collecting more data, fine-tuning and training to expect even better results.
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
- Run _**test.py**_, or you can run this block:
  ```sh 
  from test import gen_pic2cartoon
  # Choose train/test size, pic_path (input) and cartoon_path (output)
  size = TRAIN_SIZE
  pic_path = YOUR_INP_PIC_PATH
  cartoon_path  = YOUR_OUT_CARTOON_PATH
  gen_pic2cartoon(pic_path, cartoon_path, size)
  ```
- Demo input and output of this project is in _**test_result**_ dir.

