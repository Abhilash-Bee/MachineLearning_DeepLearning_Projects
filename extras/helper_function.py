# Common Dependencies
import tensorflow as tf
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import os
import random
import datetime
from sklearn.metrics import precision_recall_fscore_support, accuracy_score



# Calculates the accuracy, precision, recall, and f1-score score
def calculate_results(y_true, y_pred):
    accuracy = accuracy_score(y_true, y_pred)
    precision, recall, fscore, _ = precision_recall_fscore_support(y_true,
                                                                   y_pred,
                                                                   average='weighted')
    return {
        'Accuracy score': accuracy,
        'Precision score': precision,
        'Recall score': recall,
        'F1 score': fscore,
    }



# Plot the images with or without model prediction
def plot_images_with_or_without_prediction(dataset, class_names, model=None, prediction=False, figsize=(12, 9)):
  """
  Plot 6 random images from the dataset. If prediction = `True`, plots 6 images along with prediction
  and original label.

  Args:
  dataset - (train or test) dataset
  model - trained model, default `None`
  prediction - default `False`, if `True`, predicts with the model
  figsize - defaults to (10, 8)
  """

  rand_no = [random.randint(0, 32) for _ in range(6)]
  images = None
  labels = None

  for images, labels in dataset.take(1):
    images = images
    labels = labels

  imgs = [images[i].numpy() for i in rand_no]
  if labels[0].numpy().ndim > 0:
    lbls = [class_names[tf.argmax(labels[i])] for i in rand_no]
  else:
    lbls = [class_names[labels[i]] for i in rand_no]

  fig, ax = plt.subplots(2, 3, figsize=figsize)

  k = 0
  for i in range(2):
    for j in range(3):
      img = imgs[k]
      actual_cn = lbls[k]

      if prediction:
        y_prob = model.predict(tf.expand_dims(img, axis=0), verbose=0)
        y_prob = tf.squeeze(y_prob)
  
        if len(y_prob) > 1:
          y_pred = tf.argmax(y_prob).numpy()
        else:
          y_pred = tf.where(y_prob < 0.5, 0, 1).numpy()
  
        pred_cn = class_names[y_pred]
        if pred_cn == actual_cn:
          color = 'green'
        else:
          color = 'red'
  
        ax[i][j].set_title(f'Actual: {actual_cn}\nPredicted: {pred_cn}\nPred Prob: {tf.reduce_max(y_prob).numpy():.2f}', 
                           color=color)
      else:
        ax[i][j].set_title(actual_cn)

      img = img / 255.
      ax[i][j].imshow(img)
      
      ax[i][j].set_xticks([])
      ax[i][j].set_yticks([])

      k = k + 1



# Plot loss and accuracy curve
def plot_loss_accuracy_curve(history, figsize=(7, 10), savefig=False) -> None:
  """
  Plots the loss and accuracy curve on training and validation history.

  Args:
  history - history of the model
  figsize - defaults to `(7, 10)`
  savefig - defaults to `False`, if `True`, saves the image as `.png`
  """

  fig, ax = plt.subplots(2, 1, figsize=figsize, sharex=True)

  loss = [history.history['loss'], history.history['val_loss']]
  accuracy = [history.history['accuracy'], history.history['val_accuracy']]
  loss_accuracy = [loss, accuracy]
  labels = ['Loss', 'Accuracy']
  epoch = tf.range(1, len(history.history['loss']) + 1)

  for i in range(2):
    ax[i].set_title(labels[i] + ' Vs Epoch Curve')
    ax[i].plot(epoch, loss_accuracy[i][0], label = 'Training ' + labels[i])
    ax[i].plot(epoch, loss_accuracy[i][1], label = 'Validation ' + labels[i])
    ax[i].set_ylabel(labels[i])
    if i == 0:
      ax[i].legend(loc='upper right')
    else:
      ax[i].legend(loc='upper left')

  ax[1].set_xlabel('Epoch')

  if savefig:
    fig.savefig(f"plot_loss_accuracy_curve_{datetime.datetime.now().strftime('%Y%m%d-%H%M%S')}")



# Comparing histories of model
def compare_histories(original_history, new_history, figsize=(7, 10), savefig=False) -> None:
  """
  Plots the loss and accuracy curve on training and validation of original history
  and new history.

  Args:
  original_history - history of the original model
  new_history - history of the new model
  figsize - defaults to `(7, 10)`
  savefig - defaults to `False`, if `True`, saves the image as `.png`
  """

  tot_loss = original_history.history['loss'] + new_history.history['loss']
  tot_val_loss = original_history.history['val_loss'] + new_history.history['val_loss']

  tot_accuracy = original_history.history['accuracy'] + new_history.history['accuracy']
  tot_val_accuracy = original_history.history['val_accuracy'] + new_history.history['val_accuracy']

  labels = ['Loss', 'Accuracy']
  loss_accuracy = [(tot_loss, tot_val_loss), (tot_accuracy, tot_val_accuracy)]

  initial_epoch = len(original_history.history['loss'])

  fig, ax = plt.subplots(2, 1, figsize=figsize)

  for i in range(2):
    ax[i].set(ylabel=labels[i],
              title=labels[i]+' Vs Epoch curve')
    ax[i].plot(loss_accuracy[i][0], label='Training ' + labels[i])
    ax[i].plot(loss_accuracy[i][1], label='Validation ' + labels[i])
    ax[i].plot([initial_epoch - 1, initial_epoch - 1], plt.ylim(), label='Start Fine Tuning')
    if i == 0:
      ax[i].legend(loc='upper right')
    else:
      ax[i].legend(loc='upper left')

  ax[1].set_xlabel='Epoch'

  if savefig:
    fig.savefig(f"plot_compare_history_loss_accuracy_curve_{datetime.datetime.now().strftime('%Y%m%d-%H%M%S')}")



# Importing Dependencies
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

# Plot confusion matrix
def plot_confusion_matrix(y_true, y_pred, figsize=(10, 10), class_names=None) -> None:
  """
  Plots confusion matrix on the true and predicted label.

  Args:
  y_true - Acutal (True) Label
  y_pred - Predicted Label
  figsize - Defaults to `(10, 10)`
  class_names - Defaults to `None`, provide class_names to change x and y labels
  """

  cm = confusion_matrix(y_true=y_true, y_pred=y_pred)
  fig, ax = plt.subplots(figsize=figsize)
  disp = ConfusionMatrixDisplay(cm, display_labels=class_names)
  disp.plot(ax=ax, xticks_rotation='vertical', cmap=plt.cm.Blues)



# Importing Dependencies
import os

# Walk through the directory
def walk_through_directory(filepath: str) -> None:
  """
  Provides number of folders and number of filenames along with filepath by
  running recursively into the subdirectories of filepath.

  Args:
  filepath - path of the directory
  """

  for dirpath, dirnames, filenames in os.walk(filepath):
    print(f"There are {len(dirnames)} folders and {len(filenames)} in this '{dirpath}' directory path.")



# Create the tensorboard callback
def tensorboard_callbacks(directory: str, experiment_name: str) -> object:
  """
  Creates a tensorboard callback and provides message if the tensorboard callback
  is successfully saved in the path.

  Args:
  directory - folder name of the tensorboard callback
  experiment_name - sub-folder inside the directory of the current experiment

  Returns:
  Tensorboard object
  """

  log_dir = directory + '/' + experiment_name + '/' + datetime.datetime.now().strftime('%Y%m%d-%H%M%S')
  callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir)
  print(f'Saving the tensorboard callbacks in {log_dir}')
  return callback



# Create the tensorflow ModelCheckpoint
def tensorflow_modelcheckpoint(directory: str, experiment_name: str, str='val_acc', sbo=False):
  """
  Creates a model checkpoint and saves in the provided directory with experiment_name
  as sub-directory with another sub-directory as datetime.

  Args:
  directory - folder name of the tensorboard model checkpoint
  experiment_name - sub-folder inside the directory of the current experiment

  Returns:
  ModelCheckpoint object
  """

  filepath = directory + '/' + experiment_name + '/' + datetime.datetime.now().strftime('%Y%m%d-%H%M%S')
  print(f'Path of the checkpoint: {filepath}')
  return tf.keras.callbacks.ModelCheckpoint(filepath=filepath, 
                                            verbose=1,
                                            str=str,
                                            save_weights_only=True,
                                            save_best_only=sbo)
