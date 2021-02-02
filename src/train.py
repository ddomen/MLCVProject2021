import torch
from timeit import default_timer as timer


def metric(y_pred, y_true):
    y_pred = torch.argmax(y_pred, dim=1)
    wrong = torch.abs(y_true - y_pred)

    return torch.sum(1 - wrong).to(torch.float32)


def train(model,
          train_loader,
          optimizer,
          criterion):
    """Trains a neural network for one epoch.

    Args:
        model: the model to train.
        train_loader: the data loader containing the training data.
        optimizer: the optimizer to use to train the model.
        criterion: the loss to optimize.

    Returns:
        the loss value on the training data.
    """
    samples_train = 0
    loss_train = 0

    model.train()
    for idx_batch, (batch, labels) in enumerate(train_loader):
        optimizer.zero_grad()
        pred_labels = model(batch)

        loss = criterion(pred_labels, labels)
        loss_train += loss.item() * len(batch)
        samples_train += len(batch)

        loss.backward()
        optimizer.step()

    loss_train /= samples_train
    return loss_train


# Validate one epoch
def validate(model,
             data_loader,
             criterion,
             metric):
    """Evaluates the model.

    Args:
        model: the model to evalaute.
        data_loader: the data loader containing the validation or test data.
        criterion: the loss function.
        metric: the metric evaluation.

    Returns:
        the loss value on the validation data.
    """
    samples_val = 0
    loss_val = 0.
    acc_val = 0

    model = model.eval()
    with torch.no_grad():
        for idx_batch, (batch, labels) in enumerate(data_loader):
            pred_labels = model(batch)
            loss = criterion(pred_labels, labels)
            loss_val += loss.item() * len(batch)
            acc_val += metric(pred_labels, labels).item()

            samples_val += len(batch)

    loss_val /= samples_val
    acc_val /= samples_val

    return loss_val, acc_val


def training_loop(num_epochs,
                  optimizer,
                  model,
                  criterion,
                  metric,
                  loader_train,
                  loader_val,
                  verbose=True):
    """Executes the training loop.

        Args:
            num_epochs: the number of epochs.
            optimizer: the optimizer to use.
            model: the mode to train.
            criterion: the loss to minimize.
            metric: the evaluation metric.
            loader_train: the data loader containing the training data.
            loader_val: the data loader containing the validation data.
            verbose: if true print the value of loss.

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

    for epoch in range(1, num_epochs + 1):
        time_start = timer()
        loss_train = train(model, loader_train, optimizer, criterion)
        loss_val, acc_val = validate(model, loader_val, criterion, metric)
        time_end = timer()

        losses_values_train.append(loss_train)
        losses_values_val.append(loss_val)
        accuracy_values_val.append(acc_val)

        if verbose:
            print(f'Epoch: {epoch} '
                  f' Loss: Train = [{loss_train:.4f}] - Val = [{loss_val:.4f}] '
                  f' Accuracy: Val = [{acc_val:.4f}] '
                  f' Time one epoch (s): {(time_end - time_start):.4f} ')

    loop_end = timer()
    time_loop = loop_end - loop_start
    if verbose:
        print(f'Time for {num_epochs} epochs (s): {(time_loop):.3f}')

    return {'loss_values_train': losses_values_train,
            'loss_values_val': losses_values_val,
            'accuracy_val': accuracy_values_val,
            'time': time_loop}
