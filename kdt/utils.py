def print_lengths(sensor_data):
    for day in range(1, 4):
        print(f"Day {day}")
        for mat in range(1, 3):
            print(f"  Matrix {mat}")
            for sensor in range(0, 8):
                print(f"    Sensor {sensor}")
                for hs in range(0, 10):
                    data = sensor_data[f"day{day}"][f"mat{mat}"][sensor][hs]
                    print(f"      Heater Step {hs}, length: {len(data)}")
