# Anime StyleGAN Project (2023)

![Language](https://img.shields.io/badge/Language-Python-f2cb1b)
![Open Source](https://badges.frapsoft.com/os/v2/open-source.svg?v=103)

## Overview

This project implements a StyleGAN (Generative Adversarial Networks for image synthesis) from scratch. The StyleGAN is trained to generate anime-style images of size 64x64 using a single GPU. The training dataset consists of images of anime characters, sourced from the [Another Anime Face Dataset on Kaggle](https://www.kaggle.com/datasets/scribbless/another-anime-face-dataset/).

The project includes a graphical user interface (GUI) that allows users to easily generate anime-style images with the trained StyleGAN.

## Dependencies

Ensure you have the required dependencies installed by running the following command:

``` bash
pip install -r requirements.txt
```

This will install the necessary packages for running the StyleGAN and the GUI.

## Training

The StyleGAN model is trained based on the principles outlined in the paper [A Style-Based Generator Architecture for Generative Adversarial Networks](https://arxiv.org/pdf/1812.04948.pdf). The training data comes from the Another Anime Face Dataset.

To train the StyleGAN model:

1. Download the training dataset from [Another Anime Face Dataset](https://www.kaggle.com/datasets/scribbless/another-anime-face-dataset/).
2. Place the dataset in an appropriate directory `/data`.
3. Run the script :

```bash
python main.py
```

4. Follow the instructions

Training might take a considerable amount of time depending on your hardware capabilities.

## GUI for Image Generation

To use the GUI for generating anime-style images:

```bash
python main.py
```
Then answer `n` to the first question.
This will launch a graphical interface where you can experiment with generating anime-style images using the trained StyleGAN.

## Project Structure

- `main.py`: Script for choosing what to do.
- `generator.py`: Code for the StyleGAN generator.
- `discriminator.py`: Code for the StyleGAN discriminator.
- `configs.py`: Code for the configuration of the training.
- `utils.py`: Utility functions for data processing and model training.
- `requirements.txt`: List of dependencies.

## Generated Images Example

<p align="center">
	<img src="ressources/img/game.JPG" width="400">
</p>

<p align="center">
	<img src="ressources/img/move.PNG" width="400">
</p>

## Credits

This project was inspired by the paper [A Style-Based Generator Architecture for Generative Adversarial Networks](https://arxiv.org/pdf/1812.04948.pdf) and developed by Guillaume DI FATTA. Feel free to report issues, or provide feedback to enhance the StyleGAN image generation experience.
