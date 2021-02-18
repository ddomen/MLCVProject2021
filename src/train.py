import numpy as np
import torch
from timeit import default_timer as timer


def metric(y_pred, y_true):
    wrong = torch.abs(y_true - y_pred)
    for i in range(len(wrong)):
        if wrong[i] == -0.5 or wrong[i] == 0.5:
            wrong[i] = 1
    return torch.sum(1 - wrong).to(torch.float32)


# Train one epoch
def train(model,
          train_loader,
          idxs_train):
    """Trains a neural network for one epoch.

    Args:
        model: the model to train.
        train_loader: the data loader containing the training data.

    Returns:
        the loss value on the training data.
    """
    samples_train = 0
    loss = 0
    std = 0
    model._train()
    for idxs_batches, (batch, labels) in enumerate(train_loader):
        if idxs_batches in idxs_train:
            loss_train, std_train = model.train_step(batch, labels)
            loss += loss_train
            std += std_train
            samples_train += len(batch)

    loss /= samples_train
    std /= samples_train

    return loss, std


# Validate one epoch
def validate(model,
             data_loader,
             metric,
             idxs_val):
    """Evaluates the model.

    Args:
        model: the model to evalaute.
        data_loader: the data loader containing the validation or test data.
        criterion: the loss function.

    Returns:
        the loss value on the validation data.
    """
    samples_val = 0
    loss_val = 0.
    acc_val = 0
    number_answers = 0

    model._eval()
    with torch.no_grad():
        for idxs_batches, (batch, labels) in enumerate(data_loader):
            if idxs_batches in idxs_val:
                loss_batch, hit_batch, number_answer = model.validation_step(batch, labels, metric)
                loss_val += loss_batch
                acc_val += hit_batch
                number_answers += number_answer
                samples_val += len(batch)

    loss_val /= samples_val
    if number_answers != 0:
        acc_val /= number_answers
    number_answers /= samples_val

    return loss_val, acc_val, number_answers


def training_loop(model,
                  metric,
                  loader_train,
                  len_subset_train,
                  len_subset_validation,
                  shift,
                  verbose=True):
    """Executes the training loop.

        Args:
            num_epochs: the number of epochs.
            model: the mode to train.
            criterion: the loss to minimize.
            loader_train: the data loader containing the training data.
            verbose: if true print the value of loss.
            verbose:
            shift:
            len_subset_validation:
        Returns:
            A dictionary with the statistics computed during the train:
            the values for the train loss for each epoch.
            the values for the validation loss for each epoch.
            the time of execution in seconds for the entire loop.

    """
    global loss_train, std_train, loss_val, train_acc, acc_val, time_end, time_start

    losses_values_train = []
    losses_values_val = []
    accuracy_values_val = []
    accuracy_values_train = []
    std_values_train = []
    percent_answers_train = []
    percent_answers_val = []

    time_start = timer()
    num_batches = len(loader_train)
    iter = 0
    for i in range(0, num_batches - (len_subset_train + len_subset_validation), shift):
        idxs_train = range(i, i + len_subset_train)
        idxs_validation = range(i + len_subset_train, i + len_subset_train + len_subset_validation)
        loss_train, std_train = train(model, loader_train, idxs_train)
        loss_val, acc_val, percent_answer_train = validate(model, loader_train, metric, idxs_validation)
        _, train_acc, percent_answer_val = validate(model, loader_train, metric, idxs_train)

        losses_values_train.append(loss_train.item())
        std_values_train.append(std_train.item())
        losses_values_val.append(loss_val)
        accuracy_values_val.append(acc_val.item())
        accuracy_values_train.append(train_acc.item())
        percent_answers_train.append(percent_answer_train)
        percent_answers_val.append(percent_answer_val)

        iter += 1

    time_end = timer()
    if verbose:
        print(f'Loss: Train (mean) = [{np.mean(losses_values_train):.4f}] - Train (std) = [{np.mean(std_values_train):.4f}] - Val = [{np.mean(losses_values_val):.4f}]'
            f' Coverage (%): Train = [{np.mean(percent_answers_train):.4f}] - Val = [{np.mean(percent_answers_val):.4f}]'
            f' Accuracy: Train = [{np.mean(accuracy_values_train):.4f}] - Val = [{np.mean(accuracy_values_val):.4f}]'
            f' Time (s): {(time_end - time_start):.4f} ')

    return {'loss_values_train': np.mean(losses_values_train) * iter,
            'loss_values_val': np.mean(losses_values_val) * iter,
            'percent_answer_train': np.mean(percent_answers_train) * iter,
            'percent_answer_val': np.mean(percent_answers_val) * iter,
            'accuracy_train': np.mean(accuracy_values_train) * iter,
            'accuracy_val': np.mean(accuracy_values_val) * iter}, iter
