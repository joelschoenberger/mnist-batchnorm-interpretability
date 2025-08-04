# experiments/gradcam/gradcam_single.py
from typing import Optional
import argparse, pathlib
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from torchvision import datasets, transforms
# from torchvision.utils import make_grid  # not strictly needed, but handy
from src.models import CNN_BN

def grad_cam(
    mod: torch.nn.Module,
    x: torch.Tensor,
    target_class: Optional[int],
    layer_name: str = "bn2",
) -> torch.Tensor:
    """
    Compute a Grad-CAM heatmap for one image.

    Args:
        mod: CNN model in eval() mode, already on the right device.
        x:  input tensor, shape (1, 1, 28, 28) on the same device as `mod`.
        target_class: int class index to explain; if None use model's argmax.
        layer_name: name of the module to hook ("bn2" or "conv2").

    Returns:
        cam: heatmap tensor of shape (1, 1, 28, 28) on CPU in [0,1].
    """
    acts, grads = {}, {}

    layer = getattr(mod, layer_name)

    def _save_act(module, inp, out):
        acts["v"] = out

    def _save_grad(module, grad_input, grad_output):
        grads["v"] = grad_output[0]

    h1 = layer.register_forward_hook(_save_act)
    h2 = layer.register_full_backward_hook(_save_grad)

    out = mod(x)  # 1 x 10 (log-probabilities)

    if target_class is None:
        c = out.argmax(dim=1)
    else:
        c = torch.tensor([target_class], device=out.device, dtype=torch.long)

    loss = out[torch.arange(out.size(0), device=out.device), c].sum()

    mod.zero_grad(set_to_none=True)
    loss.backward()

    A  = acts["v"]         # (1, C, H, W)
    dA = grads["v"]        # (1, C, H, W)

    w = dA.mean(dim=(2, 3), keepdim=True)          # (1, C, 1, 1)
    cam = (w * A).sum(dim=1, keepdim=True)         # (1, 1, H, W)
    cam = F.relu(cam)
    cam = F.interpolate(cam, size=(28, 28), mode="bilinear", align_corners=False)
    cam = cam - cam.min()
    cam = cam / (cam.max() + 1e-8)

    h1.remove(); h2.remove()
    return cam.detach().cpu()

def fgsm_attack(mod: torch.nn.Module, x: torch.Tensor, y: int, eps: float) -> torch.Tensor:
    x = x.clone().detach().requires_grad_(True)
    print("minmax", x.min(), " ", x.max())
    out = mod(x)
    loss = F.nll_loss(out, torch.tensor([y], device=out.device, dtype=torch.long))
    mod.zero_grad(set_to_none=True)
    loss.backward()
    adv = (x + eps * x.grad.sign()).clamp(0, 1).detach()
    adv = (x + eps * x.grad.sign()).detach()
    return adv


def main():
    ap = argparse.ArgumentParser(description="Grad‑CAM grid: clean + FGSM eps rows")
    ap.add_argument("--count", type=int, default=5, help="number of distinct digits (columns)")
    ap.add_argument("--eps", type=float, nargs="*", default=[0.1, 0.2, 0.3],
                    help="FGSM epsilons (normalized space) for rows 2..R")
    ap.add_argument("--layer", type=str, default="bn2", choices=["bn2", "conv2"],
                    help="layer to hook for Grad‑CAM")
    ap.add_argument("--klass", dest="target_class", type=int, default=None,
                    help="explain this class instead of the predicted one")
    ap.add_argument("--cpu", action="store_true", help="force CPU")
    ap.add_argument("--stepwise", action="store_true", help="apply FGSM 100 times")
    args = ap.parse_args()

    device = torch.device("cpu" if args.cpu or not torch.cuda.is_available() else "cuda")

    # ---------- model ----------
    model = CNN_BN().to(device)
    weight_path = pathlib.Path("experiments/bn_speed/cnn_bn.pt")
    model.load_state_dict(torch.load(weight_path, map_location=device))
    model.eval()

    # ---------- data ----------
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    test_set = datasets.MNIST("data", train=False, download=True, transform=transform)

    # pick 'count' indices with distinct labels (different digits)
    wanted = args.count
    sel_indices, seen = [], set()
    i = 0
    while len(sel_indices) < wanted and i < len(test_set):
        img, lab = test_set[i]
        lab_i = int(lab)
        if lab_i not in seen:
            sel_indices.append(i)
            seen.add(lab_i)
        i += 1
    cols = len(sel_indices)
    if cols < wanted:
        print(f"Warning: only found {cols} distinct digits.")

    # rows: clean + one row per epsilon
    eps_list = [0.0] + list(args.eps)
    rows = len(eps_list)

    # storage for plotting (CPU tensors)
    images = [[None] * cols for _ in range(rows)]
    cams   = [[None] * cols for _ in range(rows)]
    titles = [[None] * cols for _ in range(rows)]

    # ---------- compute CAMs ----------
    for c, idx in enumerate(sel_indices):
        img, label = test_set[idx]
        x = img.unsqueeze(0).to(device)

        # clean
        cam_clean = grad_cam(model, x, args.target_class, layer_name=args.layer)
        with torch.no_grad():
            pred_clean = model(x).argmax(1).item()

        images[0][c] = img.unsqueeze(0).cpu()    # (1,1,28,28)
        cams[0][c]   = cam_clean                 # (1,1,28,28) on CPU
        titles[0][c] = f"true={int(label)}  pred={pred_clean}"

        # FGSM rows
        for r, eps in enumerate(eps_list[1:], start=1):
            if args.stepwise:
                adv = x
                for i in range(100):
                    adv = fgsm_attack(model, adv, int(label), eps)
            else:
                adv = fgsm_attack(model, x, int(label), eps)

            cam_adv = grad_cam(model, adv, args.target_class, layer_name=args.layer)
            with torch.no_grad():
                pred_adv = model(adv).argmax(1).item()

            images[r][c] = adv.cpu()
            cams[r][c]   = cam_adv
            titles[r][c] = f"pred={pred_adv}"

    # ---------- plotting ----------
    out_dir = pathlib.Path("experiments/gradcam/heatmaps")
    out_dir.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(rows, cols, figsize=(3.2 * cols, 3.2 * rows))

    # axes can be 1D if rows or cols == 1; normalize to 2D indexing
    if rows == 1 and cols == 1:
        axes = [[axes]]
    elif rows == 1:
        axes = [axes]
    elif cols == 1:
        axes = [[ax] for ax in axes]

    for r in range(rows):
        for c in range(cols):
            ax = axes[r][c]
            g = images[r][c].squeeze().numpy()     # grayscale
            h = cams[r][c].squeeze().numpy()       # heatmap in [0,1]
            ax.imshow(g, cmap="gray", interpolation="nearest")
            ax.imshow(h, cmap="jet", alpha=0.45, interpolation="bilinear")
            ax.set_axis_off()
            ax.set_title(titles[r][c], fontsize=9)

    # row labels on the left margin
    row_labels = ["clean"] + [f"FGSM ε={e:.2f}" for e in eps_list[1:]]
    for r in range(rows):
        axes[r][0].text(-0.15, 0.5, row_labels[r],
                        transform=axes[r][0].transAxes,
                        ha="right", va="center", rotation=90, fontsize=10)

    fig.suptitle(f"MNIST Grad‑CAM overlays", fontsize=14)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig_path = out_dir / "adv_gradcam_grid.png"
    fig.savefig(fig_path, dpi=200)
    print(f"Saved grid to {fig_path}")
    plt.show()


if __name__ == "__main__":
    main()
