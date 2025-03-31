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

* Yale (HOG): This process is shown in [(Yale-FaceRecognition)](https://github.com/chenshen03/Yale-FaceRecognition). Raw data is uploaded as "yale_hog.npy".

* Others: Shown in `Data process.ipynb`.

## Usage
We provide a GPU&CPU version for MacOS and Linux (Unknown for Windows).

Just `python TDEC.py` . 

Or submit `TDEC.sh`.

## Acknowledgement
We thank for their codes of [(DFKM)](https://github.com/hyzhang98/DFKM) that provides NN architecture.

