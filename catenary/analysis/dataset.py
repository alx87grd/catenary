import numpy as np
from abc import ABC, abstractmethod
from catenary.kinematic import powerline

# Change this for your own path
DATASET_PATH = "/Users/agirard/data/catenary/"


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

    @abstractmethod
    def outliers_count(self) -> int:
        """

        Returns the number of outliers at each frame in the dataset.

        -1 if the number of outliers is unknown.

        """

        pass


class SimulatedDataset(Dataset):
    """
    Class to generate a simulated dataset.

    dataset_params : dict
        n_out = self.params["n_out"]
        n_frames = self.params["n_frames"]
        n_obs = self.params["n_obs"]
        x_min = self.params["x_min"]
        x_max = self.params["x_max"]
        w_l = self.params["w_l"]
        w_o = self.params["w_o"]
        center = self.params["center"]
        partial_obs = self.params["partial_obs"]
        p_tru = self.params["p_tru"]
        seed = self.params["seed"]
    """

    def __init__(self, params):

        self.params = params
        name = params["name"]

        # Validate that name has format sim_[model_name]
        if not name.startswith("sim_"):
            raise ValueError("Invalid simulated dataset name")

        self.model_name = name[4:]
        self.model = powerline.create_array_model(self.model_name)

        self.generate_data()

    def generate_data(self):

        # n_frames = 100
        # n_obs = 10
        # n_out = 10
        # x_min = -5
        # x_max = 5
        # w_l = 0.2
        # w_o = 50.0
        # center = [0, 0, 0]
        # p_tru = np.array(
        #         [
        #             -22.61445006,
        #             42.86768157,
        #             14.25202579,
        #             2.31972922,
        #             698.6378392,
        #             5.83313134,
        #             7.68165757,
        #             7.28652209,
        #         ]
        #     )

        n_out = self.params["n_out"]
        n_frames = self.params["n_frames"]
        n_obs = self.params["n_obs"]
        x_min = self.params["x_min"]
        x_max = self.params["x_max"]
        w_l = self.params["w_l"]
        w_o = self.params["w_o"]
        center = self.params["center"]
        partial_obs = self.params["partial_obs"]
        p_tru = self.params["p_tru"]

        self.n_out = n_out

        # Number of lines in model
        n_lines = self.model.q

        # Initialize numpy array to store all points
        self._lidar_points = []

        for frame_idx in range(n_frames):
            # Use a different seed at each frame to randomize noise
            seed = self.params["seed"] + frame_idx
            pts = self.model.generate_test_data(
                p_tru,
                n_obs,
                x_min,
                x_max,
                w_l,
                n_out,
                center,
                w_o,
                partial_obs,
                seed,
            )

            self._lidar_points.append(pts)

    def frame_count(self) -> int:
        # Limit to 100 frames to speed up testing
        return min(self.params["n_frames"], 100)

    def lidar_points(self, idx: int):
        if idx < 0 or idx >= self.frame_count():
            raise IndexError("Index out of bounds")
        return self._lidar_points[idx]

    def filtered_lidar_points(self, idx: int):
        return self.lidar_points(idx)

    def ground_thruth_params(self, idx: int):
        if idx < 0 or idx >= self.frame_count():
            raise IndexError("Index out of bounds")

        # For now return the same ground truth for all frames
        return self.params["p_tru"]

    def outliers_count(self) -> int:

        return self.n_out


class ExperimentalDataset(Dataset):
    """
    Class to load experimental datasets.
    """

    def __init__(self, name):
        self.name = name
        self._bagfolder = DATASET_PATH + name
        self._lidar_points = np.load(self._bagfolder + "/velodyne_points.npy")
        self._filtered_lidar_points = np.load(
            self._bagfolder + "/filtered_cloud_points.npy"
        )

        # TODO: Create a function to find the ground thruth in each dataset and
        # store the result in a numpy array
        if name == "ligne315kv_test1":
            self._ground_truth = np.array(
                [
                    -22.61445006,
                    42.86768157,
                    14.25202579,
                    2.31972922,
                    698.6378392,
                    5.83313134,
                    7.68165757,
                    7.28652209,
                ]
            )
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
        return self._filtered_lidar_points[idx][
            :, ~np.isnan(self._filtered_lidar_points[idx][0])
        ]

    def ground_thruth_params(self, idx: int):
        if idx < 0 or idx >= self.frame_count():
            raise IndexError("Index out of bounds")

        # For now return the same ground truth for all frames
        return self._ground_truth

    def outliers_count(self) -> int:

        return -1


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
    if name in [
        "ligne120kv_test1",
        "ligne120kv_test2",
        "ligne120kv_test3",
        "ligne120kv_test4",
        "ligne315kv_test1",
    ]:
        return ExperimentalDataset(name)
    else:
        # TODO: Create simulated dataset
        return None


if __name__ == "__main__":

    datagen_params = {
        "name": "sim_222",
        "n_out": 10,
        "n_frames": 100,
        "n_obs": 10,
        "x_min": -50,
        "x_max": 50,
        "w_l": 0.2,
        "w_o": 50.0,
        "center": [0, 0, 0],
        "partial_obs": True,
        "p_tru": np.array(
            [
                -22.61445006,
                42.86768157,
                14.25202579,
                2.31972922,
                698.6378392,
                5.83313134,
                7.68165757,
                7.28652209,
            ]
        ),
    }

    dataset = SimulatedDataset(datagen_params)
