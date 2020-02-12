
# A simple baseline for vae

This is my codebase for the vae language model, implemented in pytorch(1.1.0):

## Overview
This package contains the code for ptb task

For the above task, the code for the following model has been made available:
1. Variational autoencoder (`vae`) 
2. Wasserstein autoencoder (`wae-det`) 
3. Generation
4. Need more time...


## Datasets
The models mentioned in the paper have been evaluated on two datasets:
 - [PTB] already in the data folder
 - [SNLI Sentences](https://nlp.stanford.edu/projects/snli/) 
 - other datasets that can be used in language model(ontonotes,iwslt,etc.)


## Requirements
- torchvision==1.1.0
- spacy
- sklearn
- matplotlib
- nltk>=3.4.5
- torchtext


## Instructions
1. Create a virtual environment using `conda`
```
conda create -n vae python=3.6
```
2. Activate virtual environment and install the required packages. 
```
source activate
conda activate vae
cd lm_wae/
pip install -r requirements.txt
```
3. Train the desired model, set configurations in the `config.conf` file. For example,
```
cd runner
python train.py 
``` 
- The model checkpoints are stored in `log/ptb/` directory, the summaries for Tensorboard are stored in `runner/runs/` directory. As training progresses, the result are dumped into `log/ptb/` directory.
- You can also see the generation result while training the model.
By default for `vae` and `wae`, sampling from latent space is carried out within one standard deviation from the mean <img src="https://latex.codecogs.com/svg.latex?\Large&space;z=\mu+\sigma\otimes\epsilon"/>. 
4. Unfinished
