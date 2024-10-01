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

    def get_sensor_cls(self, day, mat, sensor):
        s = self._sensor_classes[f"day{day}"][f"mat{mat}"][sensor]
        interpolated_data = s.get_interpolated_data()
        X = np.array([[]] * 10)
        y = []
        for cls_data in interpolated_data:
            cls = cls_data["class"]
            x = cls_data["sample_vals"]
            X = np.append(X, x, axis=1)
            cls_list = [cls] * len(cls_data["sample_times"])
            y.extend(cls_list)
        return X.T, np.array(y)
