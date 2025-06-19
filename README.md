# Model-Based Estimation of Overhead Power Lines Using LiDAR

![exp](https://github.com/user-attachments/assets/80c1c5b5-9182-404f-856c-3cd88df61dce)


# Colab exemple
https://colab.research.google.com/drive/10_0sNztttdJt7fCVf6RT6b8Ib6KUG8tr?usp=sharing

# setup
```
cd catenary
python -m venv .venv
source .venv\bin\activate
pip install numpy scipy matplotlib tqdm rosbags open3d prettytable
export PYTHONPATH=`pwd`
python experiments/rosbags_runs.py
```

# catenary

experiments/simulation_performance_analysis -> Estimation tests with simulated data

experiments/rosbags_runs -> load .npy real data (change the path in the file) 
