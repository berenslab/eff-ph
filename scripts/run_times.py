# measure run times on the Toy circle for various methods, feature dimensions, and sample sizes
# inspect results with run_times.ipynb
from utils.utils import measure_run_time, get_path

root_path = get_path("../data")

seeds = [0, 1, 2]

for sigma in [0.0]:
    _, _ = measure_run_time(dataset="toy_circle",
                            distance="euclidean",
                            dist_kwargs={},
                            embd_dim=2,
                            n=1000,
                            seeds=seeds,
                            feature_dim=1,
                            root_path=root_path,
                            sigma=sigma,
                            force_recompute=True)

    _, _ = measure_run_time(dataset="toy_circle",
                            distance="euclidean",
                            dist_kwargs={},
                            embd_dim=50,
                            n=1000,
                            seeds=seeds,
                            feature_dim=1,
                            root_path=root_path,
                            sigma=sigma,
                            force_recompute=True)

    _, _ = measure_run_time(dataset="toy_circle",
                            distance="euclidean",
                            dist_kwargs={},
                            embd_dim=50,
                            n=1000,
                            seeds=seeds,
                            feature_dim=1,
                            root_path=root_path,
                            sigma=sigma,
                            force_recompute=True)

    _, _ = measure_run_time(dataset="toy_circle",
                            distance="euclidean",
                            dist_kwargs={},
                            embd_dim=50,
                            n=2000,
                            seeds=seeds,
                            feature_dim=1,
                            root_path=root_path,
                            sigma=sigma,
                            force_recompute=True)

    _, _ = measure_run_time(dataset="toy_circle",
                            distance="eff_res",
                            dist_kwargs={"corrected": True, "weighted": False, "k": 100, "disconnect": True},
                            embd_dim=50,
                            n=1000,
                            seeds=seeds,
                            feature_dim=1,
                            root_path=root_path,
                            sigma=sigma,
                            force_recompute=True)

    _, _ = measure_run_time(dataset="toy_sphere",
                            distance="euclidean",
                            dist_kwargs={},
                            embd_dim=50,
                            n=1000,
                            seeds=seeds,
                            feature_dim=1,
                            root_path=root_path,
                            sigma=sigma,
                            force_recompute=True)

    _, _ = measure_run_time(dataset="toy_sphere",
                            distance="euclidean",
                            dist_kwargs={},
                            embd_dim=50,
                            n=1000,
                            seeds=seeds,
                            feature_dim=2,
                            root_path=root_path,
                            sigma=sigma,
                            force_recompute=True)
    print(f"Done with sigma {sigma}")