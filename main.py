from sklearn.preprocessing import OneHotEncoder
from utils import *
from sklearn.utils import shuffle
from nnet import LeNet
import tensorflow as tf
import datetime
import time
import numpy as np
import os
import cv2

BATCH_SIZE = 128
EPOCHS = 20


def evaluate(X_test, y_test, checkpoint_file):
    with tf.Session() as sess:
        # Load the saved meta graph and restore variables
        saver = tf.train.import_meta_graph("{}.meta".format(checkpoint_file))
        saver.restore(sess, checkpoint_file)

        # Get the placeholders from the graph by name
        features = sess.graph.get_operation_by_name("features").outputs[0]
        keep_prob = sess.graph.get_operation_by_name("keep_prob").outputs[0]

        # Tensors we want to evaluate
        predictions = sess.graph.get_operation_by_name("output/predictions").outputs[0]

        size = len(X_test)
        all_predictions = []
        for offset in range(0, size, BATCH_SIZE):
            end = offset + BATCH_SIZE
            if end >= size:
                end = size
            x_test_batch = X_test[offset:end]
            batch_predictions = sess.run(predictions, {features: x_test_batch, keep_prob: 1.0})
            print(batch_predictions)
            all_predictions = np.concatenate([all_predictions, batch_predictions])
        # print(all_predictions)
        print(y_test)
        correct_predictions = float(sum(all_predictions == y_test))
        print("Total number of test examples: {}".format(len(y_test)))
        print("Accuracy: {:g}".format(correct_predictions/float(len(y_test))))


def train():
    X_train, y_train, X_valid, y_valid, X_test, y_test = load_data()
    sign_list = load_signlist()
    enc = OneHotEncoder()
    X_train, X_valid, X_test = map(lambda x: normalize(grayscale(x)), [X_train, X_valid, X_test])
    y_train, y_valid, y_test =  map(lambda x: enc.fit_transform(X=x.reshape(-1,1)).toarray(), [y_train, y_valid, y_test])

    with tf.Session() as sess:

        lenet = LeNet(num_classes=43)
        num_examples = len(X_train)

        # Define Training procedure
        global_step = tf.Variable(0, name="global_step", trainable=False)
        optimizer = tf.train.AdamOptimizer(1e-3)
        grads_and_vars = optimizer.compute_gradients(lenet.loss)
        train_op = optimizer.apply_gradients(grads_and_vars, global_step=global_step)

        # Output directory for models and summaries
        timestamp = str(int(time.time()))
        out_dir = os.path.abspath(os.path.join(os.path.curdir, "runs", timestamp))
        print("Writing to {}\n".format(out_dir))

        # Summaries for loss and accuracy
        loss_summary = tf.summary.scalar("loss", lenet.loss)
        acc_summary = tf.summary.scalar("accuracy", lenet.accuracy)

        # Train Summaries
        train_summary_op = tf.summary.merge([loss_summary, acc_summary])
        train_summary_dir = os.path.join(out_dir, "summaries", "train")
        train_summary_writer = tf.summary.FileWriter(train_summary_dir, sess.graph)

        # Test Summaries
        test_summary_op = tf.summary.merge([loss_summary, acc_summary])
        test_summary_dir = os.path.join(out_dir, "summaries", "test")
        test_summary_writer = tf.summary.FileWriter(test_summary_dir, sess.graph)

        # Checkpoint directory. Tensorflow assumes this directory already exists so we need to create it
        checkpoint_dir = os.path.abspath(os.path.join(out_dir, "checkpoints"))
        checkpoint_prefix = os.path.join(checkpoint_dir, "model")
        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)
        saver = tf.train.Saver(tf.global_variables())

        sess.run(tf.global_variables_initializer())

        def train_step(x_batch, y_batch):
            """
            A single training step
            """
            feed_dict = {
              lenet.features: x_batch,
              lenet.labels: y_batch,
              lenet.keep_prob: 0.5
            }
            _, step, loss, acc, summaries = sess.run(
                [train_op, global_step, lenet.loss, lenet.accuracy, train_summary_op],
                feed_dict)
            time_str = datetime.datetime.now().isoformat()
            print("{}: step {}, loss {:g}, acc {:g}".format(time_str, step, loss, acc))
            train_summary_writer.add_summary(summaries, step)

        def test_step(x_data, y_data):
            """
            A single testing step
            """
            size = x_data.shape[0]
            total_accuray = 0
            for offset in range(0, size, BATCH_SIZE):
                end = offset + BATCH_SIZE
                if end >= size:
                    end = size
                x_batch, y_batch = x_data[offset:end], y_data[offset:end]

                feed_dict = {
                  lenet.features: x_batch,
                  lenet.labels: y_batch,
                  lenet.keep_prob: 1.0
                }
                loss, acc = sess.run(
                    [lenet.loss, lenet.accuracy],
                    feed_dict)
                total_accuray += acc * len(x_batch)
            time_str = datetime.datetime.now().isoformat()
            print("{}: acc {:g}".format(time_str, total_accuray/size))
            # test_summary_writer.add_summary(summaries, step)


        print("Training...")
        print()
        for i in range(EPOCHS):
            X_train, y_train = shuffle(X_train, y_train)
            for offset in range(0, num_examples, BATCH_SIZE):
                end = offset + BATCH_SIZE
                if end >= num_examples:
                    end = num_examples
                batch_x, batch_y = X_train[offset:end], y_train[offset:end]
                train_step(batch_x, batch_y)
            print("\nEvaluation:")
            print("="*50)
            test_step(X_valid, y_valid)
            print("="*50)
            print("Finished {} epochs".format(i+1))

        test_step(X_test, y_test)
        print("="*50)
        print("")
        current_step = tf.train.global_step(sess, global_step)
        path = saver.save(sess, checkpoint_prefix, global_step=current_step)
        print("Saved model checkpoint to {}\n".format(path))
        print("Model saved")

if __name__ == '__main__':
    # train()
    img_list = ['examples/2.jpg', 'examples/14.jpg', 'examples/17.jpg', 'examples/25.jpg', 'examples/35.jpg']
    images = np.array(list(map(lambda x: cv2.resize(cv2.imread(x),(32,32)), img_list)))
    new_data = normalize(grayscale(images))
    new_label = np.array([2,14,17,25,35])
    evaluate(new_data, new_label, 'runs/1490575550/checkpoints/model-5440')
