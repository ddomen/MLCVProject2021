import numpy as np
import torch
from timeit import default_timer as timer


def metric(y_pred, y_true):
    wrong = torch.abs(y_true - y_pred)
    repo = np.array(wrong)
    repo[repo == 1] = -1.0
    repo[repo == 0] = 1.0
    repo[repo == 0.5] = 0.0
    profit = np.sum(repo)

    for i in range(len(wrong)):
        if wrong[i] == 0.5:
            wrong[i] = 1

    return torch.sum(1 - wrong).to(torch.float32), profit


# Train one epoch
def train(model,
          train_loader,
          idxs_train):
    """Trains a neural network for one epoch.

    Args:
        model: the model to train.
        train_loader: the data loader containing the training data.

    """
    model._train()
    for idxs_batches, (batch, labels) in enumerate(train_loader):
        if idxs_batches in idxs_train:
            model.train_step(batch, labels)
    if model.method == "reinforcment":
        model.optimize_model()


# Validate one epoch
def validate(model,
             data_loader,
             metric,
             idxs_val):
    """Evaluates the model.

    Args:
        model: the model to evalaute.
        data_loader: the data loader containing the validation or test data.

    Returns:
        the loss value on the validation data.
    """
    samples_val = 0
    acc_val = 0
    number_answers = 0

    if model.method == "reinforcment":
        model.load_all_state_dict()
    model._eval()

    with torch.no_grad():
        for idxs_batches, (batch, labels) in enumerate(data_loader):
            if idxs_batches in idxs_val:
                hit_batch, number_answer, profit = model.validation_step(batch, labels, metric)
                acc_val += hit_batch
                number_answers += number_answer
                samples_val += batch.shape[0]
    if number_answers != 0:
        acc_val /= number_answers
    else:
        acc_val = torch.zeros(1, dtype=torch.float32)

    return acc_val, number_answers, samples_val, profit


def training_loop(model,
                  metric,
                  loader_train,
                  len_subset_train,
                  len_subset_validation,
                  shift,
                  verbose=True):
    """Executes the training loop.

        Args:
            model: the mode to train.
            loader_train: the data loader containing the training data.
            verbose: if true print the value of loss.
            shift:
            len_subset_validation:
        Returns:
            A dictionary with the statistics computed during the train:
            the values for the train loss for each epoch.
            the values for the validation loss for each epoch.
            the time of execution in seconds for the entire loop.

    """
    global train_acc, acc_val, time_end, time_start

    accuracy_values_val = 0
    accuracy_values_train = 0
    number_answers_train = 0
    number_answers_val = 0
    samples_train = 0
    samples_val = 0

    time_start = timer()
    num_batches = len(loader_train)
    iter = 0
    for i in range(0, num_batches - (len_subset_train + len_subset_validation), shift):
        idxs_train = range(i, i + len_subset_train)
        idxs_validation = range(i + len_subset_train, i + len_subset_train + len_subset_validation)
        train(model, loader_train, idxs_train)
        acc_val, number_answer_val, sample_val, profit = validate(model, loader_train, metric, idxs_validation)

        train_acc, number_answer_train, sample_train, _ = validate(model, loader_train, metric, idxs_train)

        model.wallet += profit
        accuracy_values_val += acc_val.item() * number_answer_val
        accuracy_values_train += train_acc.item() * number_answer_train
        number_answers_train += number_answer_train
        number_answers_val += number_answer_val
        samples_train += sample_train
        samples_val += sample_val

        iter += 1

    if number_answers_train == 0:
        accuracy_values_train = 0.0
    else:
        accuracy_values_train = accuracy_values_train / number_answers_train
    if number_answers_val == 0:
        accuracy_values_val = 0.0
    else:
        accuracy_values_val = accuracy_values_val / number_answers_val

    time_end = timer()
    if verbose:
        print(f'Coverage (%): Train = [{number_answers_train / samples_train:.4f}] - Val = [{number_answers_val / samples_val:.4f}]'
            f' Accuracy: Train = [{accuracy_values_train:.4f}] - Val = [{accuracy_values_val:.4f}]'
            f' Wallet = [{model.wallet}] - Time (s): {(time_end - time_start):.4f} ')

    return {'percent_answer_train': number_answers_train / samples_train,
            'percent_answer_val': number_answers_val / samples_val,
            'accuracy_train': accuracy_values_train * number_answers_train,
            'accuracy_val': accuracy_values_val * number_answers_val}, number_answers_val
