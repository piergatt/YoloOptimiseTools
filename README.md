# YOLOX Nano Optimisation Tools
Welcome! This repository contains a few scripts that I used during my bachelor thesis to optimise a YOLOX Nano model for inference with a neuromrophic vision sensor on an STM32F746G-DISCO board. A few scripts are also dedicated to dataset preparation. This is meant to be paired with [my other repository](https://github.com/piergatt/stm32ai-modelzoo-services-generalised_object_detection) where I show how I trained the various models generated!

## Available Uses
There are many available scripts that I used for my project. Going through the most important ones:

- [tryingOutYOLOXNano.ipynb](tryingOutYOLOXNano.ipynb) &rarr; Contains all tools for optimisation of the base st_yolo_x model found in [ST ModelZoo](https://github.com/STMicroelectronics/stm32ai-modelzoo), as well as ways to test the model

- [convertrgbtogreyscale.ipynb](convertrgbtogreyscale.ipynb) &rarr; Primarily used to convert the COCO dataset to greyscale but can be used with any RGB repository!

- [convertNumpyToImage.ipynb](convertNumpyToImage.ipynb) &rarr; Used to convert the [PEDRo](https://github.com/SSIGPRO/PEDRo-Event-Based-Dataset) dataset into usable .jpg frames

-[int4quantization.ipynb](int4quantization.ipynb) &rarr; Failed attempt at using Intels Neural Compressor to quantise a model to int4.

- [convertImageFromCamera](convertImageFromCamera) &rarr; Used to convert images from GenX320 to usable frames to test inferences in python

- [models](models) &rarr; Folder containing all models used and iterated on, subdirectory [FinalOptimisedModels](models/FinalOptimisedModels/) contains the final models used for the thesis

All other folders and files are not particularly improtant and were only used once or twice in order to satisfy my curiosity.

## How to Use

This project was tested on Python 3.10.0, with TensorFlow==2.8.4. I might create a requirements.txt later on if I have the time.