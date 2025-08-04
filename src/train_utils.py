import torch, time, random, numpy as np, tqdm

def set_seed(seed=1):
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)

def train_one_epoch(model, loader, optim, device):
    model.train()
    total, correct, loss_sum, tic = 0, 0, 0.0, time.perf_counter()
    for data, target in tqdm.tqdm(loader, leave=False): #leave=False removes progressbar afterwards
        data, target = data.to(device), target.to(device)
        optim.zero_grad()
        out   = model(data)
        loss  = torch.nn.functional.nll_loss(out, target)
        loss.backward(); optim.step()

        loss_sum += loss.item() * data.size(0)
        pred = out.argmax(1)
        correct += pred.eq(target).sum().item()
        total   += data.size(0)
    return loss_sum/total, correct/total, time.perf_counter() - tic

@torch.no_grad()
def test_epoch(model, loader, device):
    model.eval()
    total, correct, loss_sum = 0, 0, 0.0
    for data, target in loader:
        data, target = data.to(device), target.to(device)
        out  = model(data)
        loss = torch.nn.functional.nll_loss(out, target, reduction='sum')
        loss_sum += loss.item()
        pred = out.argmax(1)
        correct += pred.eq(target).sum().item()
        total   += target.size(0)
    return loss_sum/total, correct/total

# def to_rgb(t):
#     """Repeat 1‑channel tensor (N×1×H×W) into N×3×H×W float."""
#     return t.repeat(1, 3, 1, 1)
