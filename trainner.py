import tensorflow.keras as keras
import tensorflow as tf
from Model import DeepLabV3Plus
from Options import option
from DataLoader import DataLoader, configure_for_performance


class Trainner(option):
    def __init__(self):
        super(Trainner, self).__init__()

    def training(self):
        ds = DataLoader('train').load_ds()
        model = DeepLabV3Plus(256, self.num_classes).build()
        loss_fn = keras.losses.CategoricalCrossentropy(from_logits=True)
        loss_mean = keras.metrics.Mean()
        optimizer = keras.optimizers.Adam(learning_rate=self.lr)
        train_miou = keras.metrics.CategoricalAccuracy()
        # test_miou = keras.metrics.MeanIoU(num_classes=self.num_classes)

        # set tensorboard
        train_log_dir = 'logs/gradient_tape/chest_ct/first'
        train_summary_writer = tf.summary.create_file_writer(train_log_dir)

        # set checkpoint
        ckpt = tf.train.Checkpoint(step=tf.Variable(1), optimizer=optimizer, net=model)
        ckp_path = 'D:/sideproject/ckp/DeepLabv3Plus/Chest_ct/first'
        manager = tf.train.CheckpointManager(ckpt, ckp_path, max_to_keep=None)

        # print(model.summary())
        """ Training Loop """

        @tf.function
        def train_step(x, y):
            with tf.GradientTape() as tape:
                logits = model(x, training=True)
                loss = loss_fn(y, logits)

            grads = tape.gradient(loss, model.trainable_variables)
            optimizer.apply_gradients(zip(grads, model.trainable_variables))
            train_miou.update_state(y, logits)
            loss_mean.update_state(loss)

            return loss

        @tf.function
        def val_step(x, y):
            logits = model(x, training=False)
            # val_miou = test_miou(y, logits)

            # test_miou.update_state(y, val_miou)

        for ep in range(self.epoch):
            print(f'\nStart of epoch [{ep}/{self.epoch}]')

            train_ds_shuffle = configure_for_performance(ds, 1, shuffle=True)
            train_iter = iter(train_ds_shuffle)

            for step in range(self.train_cnt//self.bathSZ):
                img, mask = next(train_iter)
                loss_item = train_step(img, mask)

                if step % 10 == 0:
                    print(f'{step}/{self.train_cnt//self.bathSZ}   ||  Loss : {loss_item}  ||')

            mean_loss = loss_mean.result()
            t_miou = train_miou.result()
            manager.save()
            print(f'{ep}/{self.epoch}   ||  Loss : {mean_loss}  |   Miou : {t_miou}  ||')

            with train_summary_writer.as_default():
                tf.summary.scalar('Training Miou', t_miou, step=ep)
                tf.summary.scalar('Loss', mean_loss, step=ep)
                # tf.summary.scalar('Validation Miou', result_val_miou, step=ep)
