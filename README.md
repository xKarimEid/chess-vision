# Purpose

Train and benchmark a neural network model for ranking chess positions from winning to losing. This repository contains an implementation of a decoder-only transformer and a tokenizer. The tokenizer processes a chess position in FEN notation, and the model predicts how winning it is.

Link to the data used for training the model: [Lichess Stockfish Data](https://huggingface.co/datasets/mauricett/lichess_sf). The data contains chess positions in FEN notation and their evaluation in centipawns. In the preprocessing step, the evaluation is converted to winning percentage as explained here: [Lichess Accuracy](https://lichess.org/page/accuracy). The winning percentage is then binned as is done in the DeepMind paper: [DeepMind Paper](https://arxiv.org/abs/2402.04494).

## Steps

1. First, create a virtual environment and install the prerequisites found in the `requirements.txt` file.
   
   ```sh
   $ python -m venv venv
   $ source venv/bin/activate   # On Windows use `venv\Scripts\activate`
   $ pip install -r requirements.txt

2. Download and preprocess data from huggingface by running the following command
    ```sh
    $ python preprocess.py

3. Train the model and save the weights by running train.py
    ```sh
    $ python train.py


After training the model and saving its weights you need to wrap it in a chess engine. My chess-engine repo shows how you can do that. Then you can play against it by following my chess-ui repo.
