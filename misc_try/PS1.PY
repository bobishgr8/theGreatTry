import numpy as np
import math
from matplotlib import pyplot as plt

def MSE(pred_vector,ground_truth):
    err = 0
    n = len(pred_vector)
    for i in range(n):
        err += ((pred_vector[i]-ground_truth)**2)/n
    return err

m = [10**i for i in range(5)]
overlay_values = []
MSE_gap_vals = []
for m_value in m:
    y = np.random.randint(100)
    print(f"Train y: {y}")
    models = []
    for i in range(m_value):
        models.append(np.array([np.random.randint(100)]))

    lowest_err = float("inf")
    best_model = 0
    for i in range(len(models)):
        error = MSE(models[i],y)
        print(error)
        if error < lowest_err:
            best_model = i
            lowest_err = error

    print(f"Best model is {best_model}, with error of {lowest_err}")

    new_y = np.random.randint(100)
    print(f"Test y: {new_y}")
    test_MSE = MSE(models[best_model],new_y)
    MSE_gap_vals.append(test_MSE - lowest_err)
    overlay_values.append((math.log(m_value)/200)**0.5)

print(overlay_values)
print(MSE_gap_vals)

plt.plot(m, overlay_values)
plt.plot(m, MSE_gap_vals)

plt.show()