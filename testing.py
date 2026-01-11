import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

# Load model
model = tf.keras.models.load_model("digit_classifier_model.h5")
print("Model loaded successfully")

# Load MNIST test data
(_, _), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

# Preprocess
x_test = x_test / 255.0
x_test = x_test.reshape(-1, 28, 28, 1)

# Pick image
index = 100
image = x_test[index]
true_label = y_test[index]

# Predict
prediction = model.predict(image.reshape(1, 28, 28, 1))
predicted_digit = np.argmax(prediction)
confidence = prediction[0][predicted_digit]

# Display
plt.imshow(image.reshape(28, 28), cmap='gray')
plt.title(f"Predicted: {predicted_digit}, Actual: {true_label}")
plt.axis('off')
plt.show()

print("Confidence:", confidence)
