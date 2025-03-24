# butterfly_cnn
This is a practice/example implementation of MobileNet v3 Small trained with
ImageNet but configured to classify butterfly species native to Florida.

This code is intended to serve as a prototype to follow for the final implementation.
We will be changing many things within it when we integrate it with the hardware team's work.

MISSING DATA LOADING AND PREPROCESSING

Notes:   
        - I am running the 3.13.1 version of Python
        - Since I am focusing on implementing Fewshot, I deliberately left out the initial stages
          of the data pipeline since others are designing it
        - The file, fewshot_functions.py, contains three functions we will be using for the training of the model.
          I heavily suggest reading the paper linked in the Jupyter notebook as it will contextualize every
          operation we are performing on the data and on the model.

Goals:
        - Incorporate the prototypical model of Fewshot learning when retraining the model
        - Incorporate the preprocessing/data augmentation of the collected images from the sensors
        - Run the model on Raspberry Pi 4 in real time

