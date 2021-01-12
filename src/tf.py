import CONFIG as conf
print("loading tensorflow...")
import tensorflow as tf

print("tensorflow loaded successfully, version {}".format(tf.__version__))


def to_tf_dataset(string_tokenized, batch_size, block_size, buffer_size):
    examples = []

    for i in range(0, len(string_tokenized) - block_size + 1, block_size):
        examples.append(string_tokenized[i:i + block_size])
    inputs, labels = [], []
    for ex in examples:
        inputs.append(ex[:-1])
        labels.append(ex[1:])

    print("slicing ... ")
    dataset = tf.data.Dataset.from_tensor_slices((inputs, labels))

    print("shuffling and batching ... ")
    dataset = dataset.shuffle(buffer_size).batch(batch_size, drop_remainder=True)

    print("dataset dimension {}".format(len(dataset)))
    return dataset


def train_tf(model, tokenizer, string_tokenized,num_epoch):
    print("converting to tf dataset")

    dataset = to_tf_dataset(string_tokenized, conf.BATCH_SIZE, conf.BLOCK_SIZE, conf.BUFFER_SIZE)

    # %% [markdown]
    #
    # 2. Model Training
    # Now comes the part we’ve been waiting for, making the model and training. So we define our optimizer, loss functions and the metrics, and start training.

    # %%
    # defining our optimizer
    optimizer = tf.keras.optimizers.Adam(learning_rate=3e-5, epsilon=1e-08, clipnorm=1.0)

    # definining our loss function
    loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)

    # defining our metric which we want to observe
    metric = tf.keras.metrics.SparseCategoricalAccuracy('accuracy')

    # compiling the model
    model.compile(optimizer=optimizer, loss=[loss, *[None] * model.config.n_layer], metrics=[metric])


    cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=conf.CHECKPOINT_PATH,
                                                     save_best_only=True,
                                                     save_weights_only=True,
                                                     verbose=1)
    history = model.fit(dataset, epochs=num_epoch, callbacks=[cp_callback])


