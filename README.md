# chess-vision

# Purpose

Train and benchmark a nn model for ranking chess positions from winning to loosing

# To do 

Need to be able to run scripts from root directory

1) Create data downloading, preprocessing, creating train/test datasets

To save the files here one could convert the fen strings to ids using chatgpt encoder and then decode them
back to strings and then encode them using the chess tokenizer? 

2) Create basic tokenizer
3) Create basic neural network
4) Create predict script (Should i merge together tokenizer and model?)
5) Create training script
    - Checkpointing/saving

The workflow is the following:
    Train a nn
    Save its weights to pckle file
    Save this somewhere on the internet which enables us to load it in the chess engine repo and use it
    for making moves
