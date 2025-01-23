# setup

```
cd catenary
python -m venv .venv
source .venv\bin\activate
pip install numpy scipy matplotlib tqdm rosbags open3d
export PYTHONPATH=`pwd`
python experiments/rosbags_runs.py
```

# catenary

experiments/simulation_performance_analysis -> Estimation tests with simulated data

experiments/rosbags_runs -> load .npy real data (change the path in the file) 
