import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

class_names = [
    "T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
    "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"
]

# Load test data
(_, _), (test_images, test_labels) = tf.keras.datasets.fashion_mnist.load_data()
test_images = test_images / 255.0

# Load saved model
model = tf.keras.models.load_model("models/fashion_classifier.keras")

# Add softmax for readable probabilities
probability_model = tf.keras.Sequential([
    model,
    tf.keras.layers.Softmax()
])

index = 25
prediction = probability_model.predict(np.array([test_images[index]]))
predicted_label = np.argmax(prediction)

print(f"Predicted: {class_names[predicted_label]}")
print(f"Actual: {class_names[test_labels[index]]}")

plt.imshow(test_images[index], cmap="gray")
plt.title(f"Predicted: {class_names[predicted_label]}")
plt.axis("off")
plt.show()