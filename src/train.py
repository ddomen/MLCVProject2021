import torch
from timeit import default_timer as timer


def metric(y_pred, y_true):
    wrong = torch.abs(y_true - torch.round(y_pred))
    return torch.sum(1 - wrong).to(torch.float32)


# Train one epoch
def train(model,
          train_loader,
          criterion,
          idxs_train):
    """Trains a neural network for one epoch.

    Args:
        model: the model to train.
        train_loader: the data loader containing the training data.
        criterion: the loss to optimize.

    Returns:
        the loss value on the training data.
    """
    samples_train = 0
    loss = 0
    std = 0

    model.train()
    for idxs_batches, (batch, labels) in enumerate(train_loader):
        if idxs_batches in idxs_train:
            loss_train, std_train = model.train_step(batch, labels, criterion)
            loss += loss_train
            std += std_train
            samples_train += len(batch)

    loss /= samples_train
    std /= samples_train

    return loss, std


# Validate one epoch
def validate(model,
             data_loader,
             criterion,
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

    model.eval()
    with torch.no_grad():
        for idxs_batches, (batch, labels) in enumerate(data_loader):
            if idxs_batches in idxs_val:
                loss_batch, acc_batch = model.validation_step(batch, labels, criterion, metric)
                loss_val += loss_batch
                acc_val += acc_batch
                samples_val += len(batch)

    loss_val /= samples_val
    acc_val /= samples_val

    return loss_val, acc_val


def training_loop(num_epochs,
                  model,
                  criterion,
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
    loop_start = timer()

    losses_values_train = []
    losses_values_val = []
    accuracy_values_val = []
    accuracy_values_train = []
    std_values_train = []

    num_batches = len(loader_train)
    for i in range(0, num_batches - (len_subset_train + len_subset_validation), shift):
        idxs_train = range(i, i + len_subset_train)
        idxs_validation = range(i + len_subset_train, i + len_subset_train + len_subset_validation)
        for epoch in range(1, num_epochs + 1):
            time_start = timer()
            loss_train, std_train = train(model, loader_train, criterion, idxs_train)
            loss_val, acc_val = validate(model, loader_train, criterion, metric, idxs_validation)
            _, train_acc = validate(model, loader_train, criterion, metric, idxs_train)
            time_end = timer()

            losses_values_train.append(loss_train)
            std_values_train.append(std_train)
            losses_values_val.append(loss_val)
            accuracy_values_val.append(acc_val.item())
            accuracy_values_train.append(train_acc.item())
            if verbose:
                print(f'Epoch: {epoch} '
                      f' Loss: Train (mean) = [{loss_train:.4f}] - Train (std) = [{std_train:.4f}] - Val = [{loss_val:.4f}] '
                      f' Accuracy: Train = [{train_acc:.4f}] - Val = [{acc_val:.4f}]'
                      f' Time one epoch (s): {(time_end - time_start):.4f} ')

    loop_end = timer()
    time_loop = loop_end - loop_start

    if verbose:
        print(f'Time for {num_epochs} epochs (s): {(time_loop):.3f}')

    return {'loss_values_train': losses_values_train,
            'loss_values_val': losses_values_val,
            'accuracy_train': accuracy_values_train,
            'accuracy_val': accuracy_values_val,
            'time': time_loop}
