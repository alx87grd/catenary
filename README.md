# Model-Based Estimation of Overhead Power Lines Using LiDAR

Paper: https://arxiv.org/abs/2506.20812
Video: https://youtu.be/KMsOjzCREwk

![exp](https://github.com/user-attachments/assets/80c1c5b5-9182-404f-856c-3cd88df61dce)


# Colab exemple
https://colab.research.google.com/drive/10_0sNztttdJt7fCVf6RT6b8Ib6KUG8tr?usp=sharing

# Quick demo

Assuming numpy scipy matplotlib are already installed:
```
git clone https://github.com/alx87grd/catenary.git
cd catenary
export PYTHONPATH=`pwd`
python tests/estimator_tests.py
```

Setup for reproducing the paper results
```
git clone https://github.com/alx87grd/catenary.git
cd catenary
python -m venv .venv
source .venv\bin\activate
pip install numpy scipy matplotlib tqdm rosbags open3d prettytable
export PYTHONPATH=`pwd`
python experiments/outliers_analysis.py
```
Note: tqdm, rosbags, open3d and prettytable can be ommited for the basic functionnality. Only numpy, scipy and matplolib are required for the basic functionnality.


