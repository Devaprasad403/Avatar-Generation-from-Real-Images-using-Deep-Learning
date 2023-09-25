from tensorflow.keras.layers import Input, Conv2D, Flatten, Dense, Conv2DTranspose, Reshape
from tensorflow.keras.models import Model

# Define the encoder
latent_dim = 100
encoder_inputs = Input(shape=(128, 128, 3))
x = Conv2D(32, 3, strides=2, padding='same', activation='relu')(encoder_inputs)
x = Conv2D(64, 3, strides=2, padding='same', activation='relu')(x)
x = Flatten()(x)
latent_space = Dense(latent_dim)(x)

# Define the decoder
decoder_inputs = Input(shape=(latent_dim,))
x = Dense(32 * 32 * 64, activation='relu')(decoder_inputs)
x = Reshape((32, 32, 64))(x)
x = Conv2DTranspose(64, 3, strides=2, padding='same', activation='relu')(x)
x = Conv2DTranspose(32, 3, strides=2, padding='same', activation='relu')(x)
decoder_outputs = Conv2DTranspose(3, 3, activation='sigmoid', padding='same')(x)

# Create the VAE model
vae = Model(encoder_inputs, decoder_outputs)
