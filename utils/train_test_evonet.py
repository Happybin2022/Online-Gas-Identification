import torch

def train(net, train_loader, optimizer, criterion, param):
    net.train()
    sum_loss=0.
    i = 0
    for (x_batch, y_batch, prob_batch, patterns) in train_loader:
        optimizer.zero_grad()
        outputs, _ = net(prob_batch, y_batch, patterns)
        y_batch = torch.reshape(y_batch[:, 1:-1], (-1, param.n_event))
        loss = criterion(outputs, y_batch)
        loss.backward()
        optimizer.step()
        sum_loss += loss.item()
        i = i + 1
    loss = sum_loss / i
    return loss 

def val(net, val_loader, criterion, param):
    net.eval()
    sum_loss=0.
    i = 0
    with torch.no_grad():
        for (x_batch, y_batch, prob_batch, patterns) in val_loader:
            outputs, _ = net(prob_batch, y_batch, patterns)
            y_batch = torch.reshape(y_batch[:, 1:-1], (-1, param.n_event))
            loss = criterion(outputs, y_batch)
            sum_loss += loss.item()
            i = i + 1
        loss = sum_loss / i
    return loss

def test(net, tes_loader, criterion, param):
    net.eval()
    sum_loss=0.
    i = 0
    with torch.no_grad():
        for (x_batch, y_batch, prob_batch, patterns) in tes_loader:
            outputs, _ = net(prob_batch, y_batch, patterns)
            y_batch = torch.reshape(y_batch[:, 1:-1], (-1, param.n_event))
            loss = criterion(outputs, y_batch)
            sum_loss += loss.item()
            i = i + 1
        loss = sum_loss / i
    return loss 