import os
import time
from math import floor

import torch
from torch import nn, optim
from torch.utils.data import TensorDataset, DataLoader
from pandas import read_csv
import numpy as np
from tqdm.auto import tqdm


class EmbeddingModel(nn.Module):

    def __init__(self, embedding_dim, num_locations):
        super().__init__()

        self.embedding_dim = embedding_dim
        self.location_embed = nn.Embedding(embedding_dim=self.embedding_dim, num_embeddings=num_locations + 1)

        self.act_fn = nn.ELU()

        self.linear_1 = nn.Linear(2*embedding_dim, 100)
        self.linear_out = nn.Linear(100, 1)

    def forward(self, input):

        origin_input, destination_input = torch.chunk(input, 2, dim=-1)

        origin_embed = self.location_embed(origin_input.view(-1))
        destination_embed = self.location_embed(destination_input.view(-1))

        state_embed = torch.cat([origin_embed, destination_embed], dim=-1)
        state_embed = self.linear_1(state_embed)
        state_embed = self.act_fn(state_embed)

        output = self.linear_out(state_embed)

        return output


def get_datasets(data_path, ignore_first_row_and_col=False):
    # Get Travel Times
    travel_times = read_csv(data_path, header=None).values
    if ignore_first_row_and_col:
        travel_times = travel_times[1:,1:]
    mean_val = np.mean(travel_times)
    max_val = np.abs(travel_times).max()
    print("Mean: {}, Max: {}".format(mean_val, max_val))
    travel_times -= mean_val
    travel_times /= max_val

    num_locations = travel_times.shape[0]

    # Format
    origins = np.repeat(np.arange(1, num_locations + 1), travel_times.shape[1])
    destinations = np.tile(np.arange(1, travel_times.shape[1] + 1), num_locations)
    X = np.hstack([origins.reshape(-1, 1), destinations.reshape(-1, 1)])
    y = travel_times.reshape(-1)

    # Get train/test split
    idxs = np.array(list(range(len(y))))
    np.random.shuffle(idxs)
    train_idxs = idxs[0:floor(0.8 * len(y))]
    valid_idxs = idxs[floor(0.8 * len(y)) + 1:floor(0.9 * len(y))]
    test_idxs = idxs[floor(0.9 * len(y)) + 1:]
    X_train = torch.tensor(X[train_idxs])
    X_valid = torch.tensor(X[valid_idxs])
    X_test = torch.tensor(X[test_idxs])
    y_train = torch.tensor(y[train_idxs], dtype=torch.float32)
    y_valid = torch.tensor(y[valid_idxs], dtype=torch.float32)
    y_test = torch.tensor(y[test_idxs], dtype=torch.float32)

    train_dataset = TensorDataset(X_train, y_train)
    val_dataset = TensorDataset(X_valid, y_valid)
    test_dataset = TensorDataset(X_test, y_test)

    return train_dataset, val_dataset, test_dataset, num_locations


def fit(model, device, train_dataset, val_dataset, test_dataset, batch_size=1024, epochs=1000, patience=15, checkpoint='embedding.model', num_workers=0, use_tqdm=False):

    train_losses = []
    val_losses = []
    best_model_epoch = None
    num_epoch_digits = len(str(epochs))

    model.to(device)
    optimizer = optim.Adam(model.parameters(), eps=1e-07)
    loss_module = nn.MSELoss()

    start = time.time()

    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, pin_memory=True, num_workers=num_workers)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, num_workers=num_workers)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, num_workers=num_workers)

    for epoch in range(epochs):
        epoch_descr = f'Epoch {str(epoch + 1).zfill(num_epoch_digits)}/{epochs}'
        print(epoch_descr, end='\t')

        model.train()

        epoch_train_losses = []

        if use_tqdm:
            iterator = tqdm(train_dataloader)
        else:
            iterator = train_dataloader
        for batch_x, batch_labels in iterator:
            # Training
            batch_x = batch_x.to(device)
            batch_labels = batch_labels.to(device)
            optimizer.zero_grad()
            preds = model(batch_x)
            loss = loss_module(preds.view(-1), batch_labels)
            loss.backward()
            optimizer.step()

            # Training statistics
            epoch_train_losses.append(loss.cpu().detach().numpy())

        train_losses.append(np.mean(epoch_train_losses))

        # Train accuracy
        if use_tqdm:
            print()
        print(f'Train: {round(float(train_losses[-1]), 6): <8}', end='\t')

        # Validation
        val_loss = evaluate_model(model, device, loss_module, val_dataloader)
        val_losses.append(val_loss)
        print(f'Val: {round(float(val_loss), 6): <8}', end='\t')

        print(f'{round(time.time() - start, 1)}', end='\t')

        # Save (current) best model
        if len(val_losses) == 1 or val_loss < val_losses[best_model_epoch]:
            print('*', end='')
            best_model_epoch = epoch
            state_dict = model.state_dict()
            torch.save(state_dict, checkpoint)
        print()

        if 0 < patience < len(val_losses) and min(val_losses) < min(val_losses[-patience:]):
            print('Early stopping')
            break

        start = time.time()

    # Load best model and test it.
    state_dict = torch.load(checkpoint)
    model.load_state_dict(state_dict)
    test_loss = evaluate_model(model, device, loss_module, test_dataloader)

    print("Loss on test set: {}".format(test_loss))

    return test_loss


def evaluate_model(model, device, loss_module, dataloader):
    model.eval()

    mult_losses = []
    num_samples = 0

    for batch_x, batch_labels in dataloader:
        batch_x = batch_x.to(device)
        batch_labels = batch_labels.to(device)

        with torch.no_grad():
            preds = model(batch_x)
            loss = loss_module(preds.view(-1), batch_labels)
            mult_losses.append(loss * batch_x.shape[0])
            num_samples += batch_x.shape[0]

    loss = (torch.tensor(mult_losses) / num_samples).sum()
    return loss


def train_embeddings(embedding_dim=100, data_path='../../data/ny/zone_traveltime.csv',  base_dir='../../models/embeds/',
                     ignore_first_row_and_col=False, num_workers=0, use_tqdm=False):

    start = time.time()

    if not os.path.isdir(base_dir):
        os.makedirs(base_dir)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # Use GPU if available, else use CPU

    # get datasets
    train_dataset, val_dataset, test_dataset, num_locations = get_datasets(data_path,ignore_first_row_and_col=ignore_first_row_and_col)

    # create model
    model = EmbeddingModel(embedding_dim, num_locations)

    # Train
    test_loss = fit(
        model, device, train_dataset, val_dataset, test_dataset,
        batch_size=1024, epochs=1000, patience=15, num_workers=num_workers,
        checkpoint=os.path.join(base_dir, f'embedding_{embedding_dim}.model',),
        use_tqdm=use_tqdm
    )

    # Save Embeddings
    torch.save(model.state_dict()['location_embed.weight'], os.path.join(base_dir, f'embedding_{embedding_dim}.weights'))

    print(f'whole training took {time.time()-start}s')


if __name__ == '__main__':
    train_embeddings(num_workers=2)
