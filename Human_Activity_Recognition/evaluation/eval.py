import tensorflow as tf
from evaluation.metrics import ConfusionMatrix
import logging
import wandb
import gin

@gin.configurable
def evaluate(model_1, model_2, model_3, ds_test, ensemble, num_classes):

    metrics = ConfusionMatrix()
    accuracy_list = []

    for idx, (dataset, labels) in enumerate(ds_test):
        threshold = 0.5
        if ensemble == True:
            # Model_1
            predictions_1 = model_1(dataset, training = False)
            predictions_1 = tf.cast(predictions_1 > threshold, tf.int32)

            # Model_2
            predictions_2 = model_2(dataset, training=False)
            predictions_2 = tf.cast(predictions_2 > threshold, tf.int32)

            # Model_3
            predictions_3 = model_3(dataset, training=False)
            predictions_3 = tf.cast(predictions_3 > threshold, tf.int32)

            # final_predictions
            votes = predictions_1 + predictions_2 + predictions_3  # Count votes (0, 1, or 2, or 3)

            final_predictions = tf.cast(votes > 1, tf.int32)
        else:
            # Model_1
            final_predictions = model_1(dataset, training=False)
            #print(f"final_predictions:{final_predictions}")

        # Convert predictions to class labels
        # print(f"final_predictions(before argmax):{final_predictions}")
        predicted_labels = tf.argmax(final_predictions, axis=-1)
        true_labels = tf.cast(labels, tf.int64)
        # print(f"predicted labels: {predicted_labels}")
        # print(f"true labels: {true_labels}")

        matches = tf.cast(predicted_labels == true_labels, tf.float32)
        if tf.size(matches) > 0:
            batch_accuracy = tf.reduce_mean(matches) # tf.reduce_mean([1 2 3 4 5]) => (1+2+3+4+5)/5 = 3
            accuracy_list.append(batch_accuracy.numpy())
        else:
            print("No non-zero labels in this batch. Skipping accuracy calculation.")

        # Update confusion matrix metrics
        metrics.update_state(predicted_labels, true_labels)

    # Calculate accuracy
    accuracy = 100 * sum(accuracy_list) / len(accuracy_list)

    # Log the test accuracy to WandB
    wandb.log({'Evaluation_accuracy': accuracy})

    # Plot confusion matrix
    metrics.plot_confusion_matrix(normalize=True)

    # Results
    matrix = metrics.result().numpy()
    print(f"Evaluation_accuracy: {accuracy:6f}%")
    print("Confusion Matrix:")
    print(matrix)

    return {
        "Evaluation_accuracy": accuracy,
        "confusion_matrix": matrix
    }