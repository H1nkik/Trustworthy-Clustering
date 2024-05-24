![Static Badge](https://img.shields.io/badge/Author-H1nkik-blue)
![Static Badge](https://img.shields.io/badge/Code-Python-8A2BE2)

# -TDEC- 
## Preparation
**Enviroment**

* Python == 3.8.8

* cuda == 11.0

* torch == 1.7.1

* numpy == 1.24.3

* scikit-learn == 1.3.0

* pandas == 2.0.3

* munkres == 1.1.4 (optional)

**Data**

INPUT: dxn matrix 

* STL-10 (HOG): This process is shown in [(STL-10)]([https://pages.github.com/](https://github.com/mttk/STL10)).

* Yale (HOG): This process is shown in [(Yale-FaceRecognition)](https://github.com/chenshen03/Yale-FaceRecognition). Raw data is uploaded as "yale_hog.npy".

* Others: Shown in `Data process.ipynb`.

## Usage
We provide a GPU/CPU version for MacOS and Linux (Unknown for Windows).

Just `python TDEC.py` . 

Or submit `TDEC.sh`.
