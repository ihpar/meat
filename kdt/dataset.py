import numpy as np
from sensor_cls import SensorCls


class Dataset:
    def __init__(self, sensor_data, labels, interp_funcs):
        self._sensor_data = sensor_data
        self._labels = labels
        self._interp_funcs = interp_funcs
        self._sensor_classes = {}
        self._build_sensor_classes()

    def _build_sensor_classes(self):
        for day in range(1, 4):
            self._sensor_classes[f"day{day}"] = {}
            for mat in range(1, 3):
                self._sensor_classes[f"day{day}"][f"mat{mat}"] = []
                for sensor in range(8):
                    if day == 3 and mat == 1 and sensor == 1:
                        # Error with this sensor:
                        # missing data!
                        s = self._sensor_classes["day3"]["mat1"][0]
                    else:
                        s = SensorCls(day,
                                      mat,
                                      sensor,
                                      self._sensor_data,
                                      self._labels,
                                      self._interp_funcs)

                    self._sensor_classes[f"day{day}"][f"mat{mat}"].append(s)

    def get_sensor_cls(self, day, mat, sensor, num_samples=None, as_log=False):
        s: SensorCls = self._sensor_classes[f"day{day}"][f"mat{mat}"][sensor]
        interpolated_data = s.get_interpolated_data(
            force_num_samples=num_samples)
        X = np.array([[]] * 10)
        y = np.array([], dtype=np.int32)
        time_arr = np.array([])
        for cls_data in interpolated_data:
            X = np.append(X, cls_data["X"], axis=1)
            y = np.append(y, cls_data["y"])
            time_arr = np.append(time_arr, cls_data["time_arr"])
        if as_log:
            X = np.log(X)
        return X.T, y, time_arr

    def get_sensor_pair_cls(self, day, mat, sensor_pair,
                            num_samples=None,
                            as_log=False,
                            as_mean=False,
                            sort_by_class=False):

        if len(sensor_pair) != 2:
            raise Exception("sensors_list must contain exactly 2 sensor ids!")

        s1: SensorCls = self._sensor_classes[
            f"day{day}"][f"mat{mat}"][sensor_pair[0]]
        s2: SensorCls = self._sensor_classes[
            f"day{day}"][f"mat{mat}"][sensor_pair[1]]

        s1_data = s1.get_interpolated_data(force_num_samples=num_samples)
        s2_data = s2.get_interpolated_data(force_num_samples=num_samples)
        if sort_by_class:
            s1_data = sorted(s1_data, key=lambda d: d["class"])
            s2_data = sorted(s2_data, key=lambda d: d["class"])

        if as_mean:
            X = np.array([[]] * 10)
        else:
            X = np.array([[]] * 20)

        y = np.array([], dtype=np.int32)
        time_arr = np.array([])

        for cls_data_1, cls_data_2 in zip(s1_data, s2_data):
            if not (cls_data_1["y"] == cls_data_2["y"]).all():
                raise Exception(f"Classes are not the same!")

            if not (cls_data_1["time_arr"] == cls_data_2["time_arr"]).all():
                raise Exception(f"Time arrays are not the same!")

            X_1 = cls_data_1["X"]
            X_2 = cls_data_2["X"]
            if as_mean:
                X_1_2 = np.mean(np.array([X_1, X_2]), axis=0)
            else:
                X_1_2 = np.append(X_1, X_2, axis=0)
            X = np.append(X, X_1_2, axis=1)
            y = np.append(y, cls_data_1["y"])
            time_arr = np.append(time_arr, cls_data_1["time_arr"])

        if as_log:
            X = np.log(X)

        return X.T, y, time_arr


if __name__ == "__main__":
    import pickle
    with open("kdt/sensor_data_filtered.pkl", "rb") as f:
        sensor_data = pickle.load(f)

    with open("kdt/sensor_labels.pkl", "rb") as f:
        labels = pickle.load(f)

    with open("kdt/interpolation_functions.pkl", "rb") as f:
        interp_funcs = pickle.load(f)

    dataset = Dataset(sensor_data, labels, interp_funcs)
    X, y, time_arr = dataset.get_sensor_pair_cls(
        1, 1, (0, 1), as_mean=True)
