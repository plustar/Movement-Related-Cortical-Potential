import torch
from prefetch_generator import BackgroundGenerator

def train(model, use_cuda, train_data_loader, optim, criterion):
    model.train()
    for _, data in enumerate(BackgroundGenerator(train_data_loader)):
        img, label = data
        if use_cuda:
            img = img.cuda()
            label = label.cuda()
        optim.zero_grad()
        y_pred = model(img)
        loss = criterion(y_pred, label)
        loss.backward()
        optim.step()

def test(model, use_cuda, test_data_loader):
    model.eval()
    with torch.no_grad():
        for img, label in test_data_loader:
            if use_cuda:
                img = img.cuda()
                label = label.cuda()
            y_pred = model(img)
            pred = y_pred.argmax(dim=1, keepdim=True)
            correct = pred.eq(label.view_as(pred)).float().mean().item()
    return correct

def torch_seed_initialize():
    torch.backends.cudnn.benchmark = True
    torch.manual_seed(1)
    torch.cuda.manual_seed(1)
    torch.cuda.manual_seed_all(1)
    torch.backends.cudnn.deterministic = True