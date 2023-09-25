from tensorflow.keras.optimizers import Adam

vae.compile(optimizer=Adam(learning_rate=0.0001), loss='mse')
epochs = 50

history = vae.fit(
    train_generator,
    epochs=epochs,
    steps_per_epoch=len(train_generator),
)
