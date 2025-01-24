import numpy as np
from abc import ABC, abstractmethod

# Change this for your own path
DATASET_PATH="/home/cm9599/git/alexandre/data/"

class Dataset(ABC):
    """
    Interface class to define a dataset.
    """
    @abstractmethod
    def frame_count(self) -> int:
        """
        Returns the number of frames in the dataset.

        Returns
        -------
        int: The number of frames in the dataset.
        """
        pass

    @abstractmethod
    def lidar_points(self, idx: int) -> np.ndarray:
        """
        Returns the lidar points for a given frame.

        Parameters
        ----------
        idx : int
            The index of the frame.

        Returns
        -------
        np.ndarray
            The lidar points for the frame with shape (3, n).
        """
        pass

    @abstractmethod
    def filtered_lidar_points(self, idx: int) -> np.ndarray:
        """
        Returns the filtered lidar points for a given frame.

        Parameters
        ----------
        idx : int
            The index of the frame.

        Returns
        -------
        np.ndarray
            The filtered lidar points for the frame with shape (3, n).
        """
        pass

    @abstractmethod
    def ground_thruth_params(self, idx: int) -> np.ndarray:
        """
        Returns the ground truth parameters for a given frame.

        Parameters
        ----------
        idx : int
            The index of the frame.

        Returns
        -------
        np.ndarray
            The ground truth parameters for the frame.
        """
        pass

class ExperimentalDataset(Dataset):
    """
    Class to load experimental datasets.
    """
    def __init__(self, name):
        self.name = name
        self._bagfolder = DATASET_PATH + name
        self._lidar_points = np.load(self._bagfolder + "/velodyne_points.npy")
        self._filtered_lidar_points = np.load(self._bagfolder + "/filtered_cloud_points.npy")

        # TODO: Create a function to find the ground thruth in each dataset and
        # store the result in a numpy array
        if name == 'ligne315kv_test1':
            self._ground_truth = np.array([-22.61445006, 42.86768157, 14.25202579, 2.31972922, 698.6378392, 5.83313134, 7.68165757, 7.28652209])
        else:
            self._ground_truth = None

    def frame_count(self) -> int:
        # Limit to 100 frames to speed up testing
        return min(self._lidar_points.shape[0], 100)

    def lidar_points(self, idx: int):
        if idx < 0 or idx >= self.frame_count():
            raise IndexError("Index out of bounds")

        # Remove NaNs
        return self._lidar_points[idx][:, ~np.isnan(self._lidar_points[idx][0])]

    def filtered_lidar_points(self, idx: int):
        if idx < 0 or idx >= self.frame_count():
            raise IndexError("Index out of bounds")

        # Remove NaNs
        return self._filtered_lidar_points[idx][:, ~np.isnan(self._filtered_lidar_points[idx][0])]

    def ground_thruth_params(self, idx: int):
        if idx < 0 or idx >= self.frame_count():
            raise IndexError("Index out of bounds")

        # For now return the same ground truth for all frames
        return self._ground_truth

def load_dataset(name):
    """
    Load a dataset by name.

    Parameters
    ----------
    name : str
        The name of the dataset.

    Returns
    -------
    Dataset
        The dataset object.
    """
    if name in ["ligne120kv_test1", "ligne120kv_test2", "ligne120kv_test3", "ligne120kv_test4", "ligne315kv_test1"]:
        return ExperimentalDataset(name)
    else:
        # TODO: Create simulated dataset
        return None
