import torch
import time
from tqdm import tqdm

def train_model(model, data, epochs, optimizer, criterion, num_docs, model_dir, patience, device):
    best_val_loss = float("inf")
    best_state = None
    wait = 0
    start_time = time.time()

    for epoch in tqdm(range(1, epochs + 1), desc="Epochs"):
        model.train()
        optimizer.zero_grad()
        out = model(data.x, data.edge_index, data.edge_weight)
        loss = criterion(out[:num_docs], data.y.to(device))
        loss.backward()
        optimizer.step()

        # Validation (reuse training mask here)
        model.eval()
        with torch.no_grad():
            out_val = model(data.x, data.edge_index, data.edge_weight)
            val_loss = criterion(out_val[:num_docs], data.y.to(device)).item()

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = model.state_dict()
            torch.save(best_state, f"{model_dir}/textgcn_best.pt")
            wait = 0
        else:
            wait += 1
            if wait >= patience:
                print(f"⏹️ Early stopping at epoch {epoch}. Best loss {best_val_loss:.4f}")
                break

        if epoch % 10 == 0:
            print(f"Epoch {epoch:03d} | Train Loss={loss:.4f} | Val Loss={val_loss:.4f}")

    model.load_state_dict(best_state)
    print(f"🏁 Training done in {(time.time() - start_time)/60:.2f} min. Best ValLoss={best_val_loss:.4f}")
    return model
