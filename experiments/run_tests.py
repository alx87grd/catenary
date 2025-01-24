from experiments.test_runner import run_test, plot_results
import numpy as np
from prettytable import PrettyTable

def run_experimental_tests():
    exp_315kv_test1_params = {
        "name": "315kV Test 1",
        "dataset": "ligne315kv_test1",
        "model": "222",
        "p_0": np.array([-25.0, 40.0, 0.0, 1.0, 700, 6.0, 6.0, 6.0]),
        "Q": np.diag([0.02, 0.02, 0.002, 0.01, 0.0001, 0.02, 0.02, 0.02]),
        "l": 1.0,
        "b": 1000.0,
        "power": 2.0,
        "p_ub": np.array([200.0, 200.0, 25.0, 2.5, 1000.0, 7.0, 9.0, 9.0]),
        "p_lb": np.array([-200.0, -200.0, 0.0, 1.5, 5.0, 5.0, 6.0, 6.0]),
        "n_search": 2,
        "p_var": np.array([50.0, 50.0, 50.0, 5.0, 200.0, 2.0, 2.0, 2.0]),
        "filter_method": "",
    }

    table = PrettyTable()
    table.field_names = ["Test name", "Average num. points", "Vertex point mean error +/- std dev (x,y,z) [m]"]
    for filter_method in ["none", "ground_filter", "clustering", "corridor"]:
        params = exp_315kv_test1_params.copy()
        params["filter_method"] = filter_method
        params["name"] = f"315kV Test 1 ({filter_method})"

        results = run_test(params)

        plot_results(params, results)

        table.add_row([params["name"],
                          f'{results["num_points_mean_after_filter"]:.0f} +/- {results["num_points_std_after_filter"]:.0f}',
                            f'({results["p_err_mean"][0]:.2f}, {results["p_err_mean"][1]:.2f}, {results["p_err_mean"][2]:.2f}) +/- ' +
                                f'({results["p_err_std"][0]:.2f}, {results["p_err_std"][1]:.2f}, {results["p_err_std"][2]:.2f})'])

    print(table)

if __name__ == "__main__":
    run_experimental_tests()
