# Robust language model for noisy biomedical data
Carnegie Mellon Univ - AI for Social Good (advised Prof. Fei Fang)

"A noisy-robust language model for biomedical keyword identification with external knowledge-based data augmentation"

Support datsaset usage of NBME(the National Board of Medical Examiners)



## Install Dependencies
A set of dependencies is listed in environment.yml. You can use conda to create and activate the environment easily.
- Install appropriate torch version for your cuda version

```
conda env create -f environment.yml
conda activate deberta
```

## models
- DeBERTa
- RoBERTa
- bioBERT
- biomegatron

