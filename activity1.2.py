import numpy as np
import matplotlib.pyplot as plt

def relu(x):
    return np.maximum(0, x)

def leaky_relu(x, alpha=0.01):
    return np.where(x > 0, x, alpha * x)

def tanh(x):
    return np.tanh(x)

random_values = [-3.5, -1.2, 0, 2.8, -4.1, 1.5, -0.7, 3.2, -2.4, 4.6]

relu_values = relu(np.array(random_values))
leaky_relu_values = leaky_relu(np.array(random_values))
tanh_values = tanh(np.array(random_values))

print("ReLU values for the given data:")
for i, val in enumerate(random_values):
    print(f"Value: {val}, ReLU: {relu_values[i]}")

print("\nLeaky ReLU values for the given data:")
for i, val in enumerate(random_values):
    print(f"Value: {val}, Leaky ReLU: {leaky_relu_values[i]}")

print("\nTanh values for the given data:")
for i, val in enumerate(random_values):
    print(f"Value: {val}, Tanh: {tanh_values[i]}")

x = np.linspace(-5, 5, 100)
relu_y = relu(x)
leaky_relu_y = leaky_relu(x)
tanh_y = tanh(x)

plt.figure(figsize=(10, 6))
plt.plot(x, relu_y, label='ReLU', color='blue')
plt.plot(x, leaky_relu_y, label='Leaky ReLU', color='green')
plt.plot(x, tanh_y, label='Tanh', color='red')

plt.scatter(random_values, relu_values, color='blue', label='Data (ReLU)', zorder=5)
plt.scatter(random_values, leaky_relu_values, color='green', label='Data (Leaky ReLU)', zorder=5)
plt.scatter(random_values, tanh_values, color='red', label='Data (Tanh)', zorder=5)

plt.title('Activation Functions')
plt.xlabel('x')
plt.ylabel('y')
plt.legend()
plt.grid(True)
plt.show()

