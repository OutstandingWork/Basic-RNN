import tensorflow as tf
import numpy as np 

def generate_time_series(batch_size, n_steps):
    freq1, freq2, offsets1, offsets2 = np.random.rand(4, batch_size, 1)
    time = np.linspace(0, 1, n_steps)
    series = 0.5 * np.sin((time - offsets1) * (freq1 * 10 + 10))  #   wave 1
    series += 0.2 * np.sin((time - offsets2) * (freq2 * 20 + 20)) # + wave 2
    series += 0.1 * (np.random.rand(batch_size, n_steps) - 0.5)   # + noise
    return series[..., np.newaxis].astype(np.float32)


n_steps = 50
series = generate_time_series(10000, n_steps + 1)
X_train, y_train = series[:7000, :n_steps], series[:7000, -1]
X_valid, y_valid = series[7000:9000, :n_steps], series[7000:9000, -1]
X_test, y_test = series[9000:, :n_steps], series[9000:, -1]

model = tf.keras.models.Sequential([
 tf.keras.layers.Flatten(input_shape=[50, 1]),
 tf.keras.layers.Dense(1)])


model.compile(loss="mse", optimizer="adam")
history = model.fit(X_train, y_train, epochs=5,
 validation_data=(X_valid, y_valid))
model.evaluate(X_test, y_test)
y_pred = model.predict(X_test)
accuracy = tf.keras.metrics.Accuracy(y_test, y_pred)
print("Accuracy:", accuracy)
