# Word Embedding Dimension Reduction via Weakly-Supervised Feature Selection
Created by Jintang Xue, Yun-Cheng Wang, Chengwei Wei and C-C Jay Kuo from University of Southern California.

### Introduction
This work is an official implementation of our [paper](https://arxiv.org/abs/2407.12342). We proposed an an efficient and effective weakly-supervised feature selection method named WordFS method for word embedding dimensionality reduction.
    
### Installation

The code has been tested with Python 3.8. You may need to install the environment first.

```shell
conda env create --file environment.yml
conda activate WordFS
```

### Pre-trained word embeddings

We test our model on three different pre-trained word embeddings: 

Glove: https://nlp.stanford.edu/data/glove.6B.zip.
Word2vec: https://drive.google.com/file/d/0B7XkCwpI5KDYNlNUTTlSS21pQmM/edit?usp=sharing
Fasttext: https://dl.fbaipublicfiles.com/fasttext/vectors-english/wiki-news-300d-1M.vec.zip

### Usage
To test the model on each word similarity dataset:

    python all_dataset_main.py

To test the model on the combined word similarity dataset:

    python agg_dataset_main.py

The reduced embedding files will be saved to the ```test_wv``` folder.

The downstream tasks evalution is based on the SentEval (https://github.com/facebookresearch/SentEval). Directly follow the SentEval bow.py example with the reduced embedding file.

### Citation
If you find our work useful in your research, please consider citing:

    @article{xue2024word,
      title={Word Embedding Dimension Reduction via Weakly-Supervised Feature Selection},
      author={Xue, Jintang and Wang, Yun-Cheng and Wei, Chengwei and Kuo, C-C Jay},
      journal={arXiv preprint arXiv:2407.12342},
      year={2024}
    }

If you have any questions, please contact jintangx@usc.edu.
