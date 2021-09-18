from data_loader import Data_Loader
import os
from transformer import Transformer, Mask
import tensorflow as tf
import datetime
import time

CHECKPOINT_DIR = "./checkpoints"
D_MODEL = 200  # Dimension of embedding for model
VOCAB_SIZE = 32000
BATCH_SIZE = 32
# Take this many sentences from the data for the whole training and validation
DATA_LIMIT = 20000
D_FF = 400
DROPOUT = 0.1
ENCODER_COUNT = 4
DECODER_COUNT = 4
N_H = 4  # (number of heads, keep it divisible by D_MODEL)
EPOCHS = 2
DATA_DIR = "./data"


class MyLRSchedule(tf.keras.optimizers.schedules.LearningRateSchedule):
    def __init__(self, dmodel):
        self.dmodel = dmodel
        self.warmup_steps = 4000

    def __call__(self, step):
        arg1 = tf.math.rsqrt(step)
        arg2 = step * (self.warmup_steps ** -1.5)
        return tf.math.sqrt(tf.cast(self.dmodel, tf.float32)) * tf.math.minimum(arg1, arg2)


def train_step(source_sent, target_sent, transformer, optimiser, train_loss, train_accuracy, loss_func):
    input_target_sent = target_sent[:, :-1]
    output_target_sent = target_sent[:, 1:]
    y = tf.one_hot(output_target_sent, VOCAB_SIZE)
    encoder_padding_mask, look_ahead_mask, decoder_padding_mask = Mask.create_all_mask(
        seq_len=input_target_sent.shape[1], encoder_sequences=source_sent, decoder_sequence=input_target_sent)
    with tf.GradientTape() as tape:
        pred = transformer.call(source_sent, encoder_padding_mask,
                                decoder_padding_mask, look_ahead_mask, input_target_sent, training=True)
        loss_value = loss_func(y, pred)
    grads = tape.gradient(loss_value, transformer.trainable_weights)
    optimiser.apply_gradients(zip(grads, transformer.trainable_weights))
    train_loss(loss_value)
    train_accuracy(output_target_sent, pred)

    return tf.reduce_mean(loss_value)


def validation_step(source_sent, target_sent, transformer, validation_loss, validation_accuracy, loss_func):
    input_target_sent = target_sent[:, :-1]
    output_target_sent = target_sent[:, 1:]
    y = tf.one_hot(output_target_sent, VOCAB_SIZE)
    encoder_padding_mask, look_ahead_mask, decoder_padding_mask = Mask.create_all_mask(
        seq_len=input_target_sent.shape[1], encoder_sequences=source_sent, decoder_sequence=input_target_sent)
    with tf.GradientTape() as tape:
        pred = transformer.call(source_sent, encoder_padding_mask,
                                decoder_padding_mask, look_ahead_mask, input_target_sent, training=False)
        loss_value = loss_func(y, pred)
    validation_loss(loss_value)
    validation_accuracy(output_target_sent, pred)

    return tf.reduce_mean(loss_value)


def trainer(dataloader, val_dataloader, transformer, optimiser, loss_func):
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)

    # Create checkpoint object and its corresponding manager object
    checkpoint = tf.train.Checkpoint(step=tf.Variable(
        1), optimizer=optimiser, model=transformer)
    checkpoint_manager = tf.train.CheckpointManager(
        checkpoint, CHECKPOINT_DIR, max_to_keep=3)

    # To save all the losses and accuracy
    train_loss = tf.keras.metrics.Mean('train_loss', dtype=tf.float32)
    train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(
        'train_accuracy')
    validation_loss = tf.keras.metrics.Mean(
        'validation_loss', dtype=tf.float32)
    validation_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(
        'validation_accuracy')

    for epoch in range(EPOCHS):
        print("\nStart of epoch %d" % (epoch,))
        start_time = time.time()
        # Iterate over the batches of the dataset.
        for step, (source_sent, target_sent) in enumerate(dataloader):
            loss_value = train_step(
                source_sent, target_sent, transformer, optimiser, train_loss, train_accuracy, loss_func)
            checkpoint.step.assign_add(1)
            if(step % 100 == 0):
                print(
                    "Training loss (for one batch) at step %d: %.4f"
                    % (step, float(loss_value))
                )
                print("Seen so far: %d samples" % ((step + 1) * BATCH_SIZE))
            if int(checkpoint.step) % 400 == 0:
                save_path = checkpoint_manager.save()
                print("Saved checkpoint for step {}: {}".format(
                    int(checkpoint.step), save_path))
                print("loss {:1.2f}".format(loss_value.numpy()))

        print("{} | After Epoch: {} Loss:{}, Accuracy: {}, time: {} sec".format(
            datetime.datetime.now(), epoch, train_loss.result(), train_accuracy.result(),
            time.time() - start_time
        ))

        val_start_time = time.time()
        for step, (source_sent, target_sent) in enumerate(val_dataloader):
            loss_value = validation_step(
                source_sent, target_sent, transformer, validation_loss, validation_accuracy, loss_func)

        print("{} *** Validation Metric For Epoch: {} |Loss:{}, Accuracy: {}, time: {} sec".format(
            datetime.datetime.now(), epoch, validation_loss.result(), validation_accuracy.result(),
            time.time() - val_start_time
        ))

        train_loss.reset_states()
        train_accuracy.reset_states()
        validation_loss.reset_states()
        validation_accuracy.reset_states()

    save_path = checkpoint_manager.save()
    print("saved at the end")

    return "DONE"


def main():
    os.makedirs(DATA_DIR, exist_ok=True)
    data_loader = Data_Loader(DATA_DIR, 'wmt14/en-de',
                              data_limit=DATA_LIMIT, batch_size=BATCH_SIZE)
    dataloader, val_dataloader = data_loader.load()

    transformer = Transformer(vocab_size=VOCAB_SIZE, d_model=D_MODEL, n_h=N_H,
                              d_ff=D_FF, dropout=DROPOUT, encoder_count=ENCODER_COUNT, decoder_count=DECODER_COUNT)

    optimiser = tf.optimizers.Adam(learning_rate=MyLRSchedule(
        D_MODEL), beta_1=0.9, beta_2=0.98, epsilon=1e-9)
    loss_func = tf.losses.CategoricalCrossentropy(
        from_logits=False, label_smoothing=0.1)

    trainer(dataloader, val_dataloader, transformer, optimiser, loss_func)

    print("*******TRAINING COMPLETE*********")


if __name__ == "__main__":
    main()
