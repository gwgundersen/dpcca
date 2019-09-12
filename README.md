# Deep probabilistic CCA

Code for [End-to-end training of deep probabilistic CCA on paired biomedical observations](http://auai.org/uai2019/proceedings/papers/340.pdf). 

### Abstract

Medical pathology images are visually evaluated by experts for disease diagnosis, but the connection between image features and the state of the cells in an image is typically unknown. To understand this relationship, we develop a multimodal modeling and inference framework that estimates shared latent structure of joint gene expression levels and medical image features. Our method is built around probabilistic canonical correlation analysis (PCCA), which is fit to image embeddings that are learned using convolutional neural networks and linear embeddings of paired gene expression data. Using a differentiable take on the EM algorithm, we train the model end-to-end so that the PCCA and neural network parameters are estimated simultaneously. We demonstrate the utility of this method in constructing image features that are predictive of gene expression levels on simulated data and the Genotype-Tissue Expression data. We demonstrate that the latent variables are interpretable by disentangling the latent subspace through shared and modality-specific views.

### Installation

Install the dependencies using [Conda](https://conda.io/en/latest/) and activate the environment:

```bash
conda env create -f environment.yml
source activate dpcca
```

[Optional] Run the unit tests. Note that these occasionally fail due to numerical tolerances:

```bash
bash run_tests.sh
```

### Reproducing multimodal MNIST results

Generate the multimodal MNIST data set.

```bash
python -m data.mnist.generate
```

Create directories for experiments:

```bash
mkdir experiments experiments/example
```

Run the code:

```python
python traindpcca.py
```