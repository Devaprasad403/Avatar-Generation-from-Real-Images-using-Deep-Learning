import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.applications import VGG19
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.vgg19 import preprocess_input
from tensorflow.keras.models import Model

# Load a base image and a style image
base_image_path = 'path_to_base_image.jpg'
style_reference_image_path = 'path_to_style_image.jpg'

# Define a function to preprocess the images
def preprocess_image(image_path):
    img = image.load_img(image_path, target_size=(224, 224))
    img = image.img_to_array(img)
    img = np.expand_dims(img, axis=0)
    img = preprocess_input(img)
    return tf.convert_to_tensor(img)

# Load the base and style images
base_image = preprocess_image(base_image_path)
style_reference_image = preprocess_image(style_reference_image_path)

# Create a VGG19 model with pretrained weights (excluding the fully connected layers)
model = VGG19(weights='imagenet', include_top=False)

# Specify the layers for style and content representations
style_layers = ['block1_conv1', 'block2_conv1', 'block3_conv1', 'block4_conv1', 'block5_conv1']
content_layer = 'block4_conv2'

# Create a model that computes the style and content features
style_outputs = [model.get_layer(name).output for name in style_layers]
content_outputs = model.get_layer(content_layer).output

feature_extractor = Model(inputs=model.input, outputs=[style_outputs, content_outputs])

# Define a function to compute the Gram matrix (used for style loss)
def gram_matrix(x):
    x = tf.transpose(x, (0, 3, 1, 2))
    features = tf.reshape(x, (tf.shape(x)[0], -1))
    gram = tf.matmul(features, tf.transpose(features))
    return gram

# Define the style loss function
def style_loss(style, generated):
    style_gram = gram_matrix(style)
    generated_gram = gram_matrix(generated)
    loss = tf.reduce_mean(tf.square(style_gram - generated_gram))
    return loss

# Define the content loss function
def content_loss(content, generated):
    loss = tf.reduce_mean(tf.square(content - generated))
    return loss

# Create a tf.Variable to store the generated image
generated_image = tf.Variable(base_image)

# Define the total variation loss to reduce noise
def total_variation_loss(x):
    a = tf.square(x[:, :-1, :-1, :] - x[:, 1:, :-1, :])
    b = tf.square(x[:, :-1, :-1, :] - x[:, :-1, 1:, :])
    return tf.reduce_sum(tf.pow(a + b, 1.25))

# Define the overall loss function
def compute_loss(base_image, style_reference_image, generated_image):
    style_outputs, content_outputs = feature_extractor(generated_image)
    style_loss_value = 0.0
    content_loss_value = 0.0
    for target, gen in zip(style_reference_image, style_outputs):
        style_loss_value += style_loss(target, gen)
    style_loss_value /= len(style_outputs)
    content_loss_value = content_loss(content_outputs[0], content_outputs[1])
    total_variation_loss_value = total_variation_loss(generated_image)
    alpha = 1e-2
    beta = 1e2
    total_loss = alpha * content_loss_value + beta * style_loss_value + total_variation_loss_value
    return total_loss

# Define the optimizer
opt = tf.optimizers.Adam(learning_rate=0.02, beta_1=0.99, epsilon=1e-1)

# Training loop
epochs = 10
for epoch in range(epochs):
    with tf.GradientTape() as tape:
        loss = compute_loss(base_image, style_reference_image, generated_image)
    grads = tape.gradient(loss, generated_image)
    opt.apply_gradients([(grads, generated_image)])
    generated_image.assign(tf.clip_by_value(generated_image, clip_value_min=0.0, clip_value_max=1.0))

# Display the generated image
plt.imshow(tf.squeeze(generated_image))
plt.axis('off')
plt.show()
