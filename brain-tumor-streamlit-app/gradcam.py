"""Grad-CAM utilities for visualizing which MRI region drove a prediction."""
import numpy as np
import tensorflow as tf
from matplotlib import cm
from PIL import Image


def find_last_conv_layer(model):
    for layer in reversed(model.layers):
        if isinstance(layer, tf.keras.Model):
            try:
                return find_last_conv_layer(layer)
            except ValueError:
                continue
        if isinstance(layer, tf.keras.layers.Conv2D):
            return layer.name
    raise ValueError("No convolutional layer found in model.")


def make_gradcam_heatmap(img_array, model, last_conv_layer_name=None, pred_index=None):
    if last_conv_layer_name is None:
        last_conv_layer_name = find_last_conv_layer(model)

    base_model = None
    for layer in model.layers:
        if isinstance(layer, tf.keras.Model) and layer.get_layer(last_conv_layer_name, None) is not None:
            base_model = layer
            break

    if base_model is not None:
        conv_layer = base_model.get_layer(last_conv_layer_name)
        conv_model = tf.keras.Model(base_model.inputs, conv_layer.output)

        classifier_input = tf.keras.Input(shape=conv_layer.output.shape[1:])
        x = classifier_input
        start = model.layers.index(base_model) + 1
        for layer in model.layers[start:]:
            x = layer(x)
        classifier_model = tf.keras.Model(classifier_input, x)

        with tf.GradientTape() as tape:
            conv_output = conv_model(img_array)
            tape.watch(conv_output)
            preds = classifier_model(conv_output)
            if pred_index is None:
                pred_index = tf.argmax(preds[0])
            class_channel = preds[:, pred_index]
        grads = tape.gradient(class_channel, conv_output)
    else:
        grad_model = tf.keras.Model(model.inputs, [model.get_layer(last_conv_layer_name).output, model.output])
        with tf.GradientTape() as tape:
            conv_output, preds = grad_model(img_array)
            if pred_index is None:
                pred_index = tf.argmax(preds[0])
            class_channel = preds[:, pred_index]
        grads = tape.gradient(class_channel, conv_output)

    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    conv_output = conv_output[0]
    heatmap = conv_output @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)
    heatmap = tf.maximum(heatmap, 0) / (tf.math.reduce_max(heatmap) + 1e-8)
    return heatmap.numpy()


def overlay_heatmap(original_image: Image.Image, heatmap: np.ndarray, alpha: float = 0.4) -> Image.Image:
    heatmap_resized = Image.fromarray(np.uint8(255 * heatmap)).resize(original_image.size)
    colored = cm.jet(np.array(heatmap_resized))[:, :, :3]
    colored = Image.fromarray(np.uint8(colored * 255))
    return Image.blend(original_image.convert("RGB"), colored, alpha)
