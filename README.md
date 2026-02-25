![Static Badge](https://img.shields.io/badge/Deep_Clustering-blue)
![Static Badge](https://img.shields.io/badge/Code-PyTorch-8A2BE2)




# -TDCC- 
"All models are wrong, but some — that know when they can be trusted — are useful!"
                                                             ——George E.P. Box (Adapted)
## Preparation
**Dependency**

* Python == 3.8.8

* cuda == 11.0

* torch == 1.7.1

* numpy == 1.24.3

* scikit-learn == 1.3.0

* pandas == 2.0.3

* munkres == 1.1.4 (optional)

**Data**

INPUT: dxn matrix 

* STL-10 (HOG): This process is shown in [(STL-10)](https://github.com/mttk/STL10).

* Yale (HOG): This process is shown in [(Yale-FaceRecognition)](https://github.com/chenshen03/Yale-FaceRecognition).

* HHAR: In [(HHAR)](https://github.com/yueliu1999/Awesome-Deep-Graph-Clustering/tree/main/dataset).

* Others: Shown in `Data process.ipynb`.

**Ablation**

Please see `TDCC-SSE.ipynb`. The Gaussian datasets of Machine Learning Group (University of Eastern Finland) is here [(G2)](http://cs.uef.fi/sipu/datasets/).

## Usage
We provide a GPU&CPU version for MacOS and Linux (Unknown for Windows).

Just `python TDCC.py` . 

Or submit `TDCC.sh`.

## Acknowledgement
We thank for their codes of [(DFKM)](https://github.com/hyzhang98/DFKM) that provides NN architecture.

## Citation

If you find MvWECM useful in your research, please consider citing:

**BibTeX**
```bibtex
@article{zhu2026,
  title = {TDCC: A Trustworthy Deep Credal Clustering Method for Uncertain Data},
  author = {Yuchen Zhu and Kuang Zhou and Fabio Cuzzolin},
  journal = {IEEE Transactions on Cybernetics},
  year = {2026},
  volume = {},
  number = {},
  pages = {}
}


