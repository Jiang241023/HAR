import tensorflow as tf
from evaluation.metrics import ConfusionMatrix
import logging
import wandb
import gin

@gin.configurable
def evaluate(model_1, model_2, ds_test, ensemble):

    metrics = ConfusionMatrix()
    accuracy_list = []

    for idx, (dataset, labels) in enumerate(ds_test):
        if ensemble == True:
            # Model_1
            predictions_1 = model_1(dataset, training = False)
            predicted_labels_1 = tf.argmax(predictions_1, axis=-1)  # Class labels


            # Model_2
            predictions_2 = model_2(dataset, training=False)
            predicted_labels_2 = tf.argmax(predictions_2, axis=-1)  # Class labels

            # final_predictions
            votes = tf.stack([predicted_labels_1, predicted_labels_2], axis=1)  # Shape: (batch_size, 2)
            print(f"votes:{votes}")
            final_predictions = tf.reduce_sum(votes, axis=1) // 2  # Majority vote
            print(f"final_predictions:{final_predictions}")
        else:
            # Model_1
            final_predictions = model_2(dataset, training=False)
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