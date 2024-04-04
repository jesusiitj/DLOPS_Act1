import numpy as np
import matplotlib.pyplot as plt

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

random_values = [-3.5, -1.2, 0, 2.8, -4.1, 1.5, -0.7, 3.2, -2.4, 4.6]

sigmoid_values = sigmoid(np.array(random_values))

print("Sigmoid values for the given data:")
for i, val in enumerate(random_values):
    print(f"Value: {val}, Sigmoid: {sigmoid_values[i]}")

x = np.linspace(-5, 5, 100)
y = sigmoid(x)

plt.figure(figsize=(8, 5))
plt.plot(x, y, label='Sigmoid')
plt.scatter(random_values, sigmoid_values, color='red', label='Data Points')
plt.title('Sigmoid Function')
plt.xlabel('x')
plt.ylabel('Sigmoid(x)')
plt.legend()
plt.grid(True)
plt.show()