import os
import numpy as np
from PIL import Image

# WRITING SHALLOW ANN FROM SCRATCH (AI DEBUGGED, WITH LINE SEARCH)

class ANN:
    def __init__(self):
        # HIDDEN LAYER
        rng = np.random.default_rng(seed=42)
        self.w = (rng.random(size=(10, 12600)) - 0.5) * 0.01
        self.b = np.zeros((10, 1))

        # OUTPUT LAYER
        self.w2 = (rng.random(size=(10, 10)) - 0.5) * 0.01
        self.b2 = np.zeros((10, 1))
    
    def relu(self, z):
        return np.maximum(0, z)
    
    def drelu(self, z):
        return (z > 0).astype(float)
    
    def forward(self, x):
        self.x = x.T                              
        self.z = (self.w @ self.x) + self.b        
        self.a = self.relu(self.z)                 

        self.z2 = (self.w2 @ self.a) + self.b2    
        return self.z2
    
    def cross_entropy_loss(self, z2, ytrue, return_softmax=False):
        n = z2.shape[1]                            
        
        shifted = z2 - np.max(z2, axis=0, keepdims=True)
        softmax = np.exp(shifted) / np.sum(np.exp(shifted), axis=0, keepdims=True)
        
        probs = np.clip(softmax[ytrue, np.arange(n)], 1e-15, 1.0)
        celoss = -np.log(probs)
        loss = np.mean(celoss)

        if return_softmax:
            return loss, softmax
        return loss

    def gradients(self, z2, ytrue):
        n = z2.shape[1]
        loss, softmax = self.cross_entropy_loss(z2, ytrue, return_softmax=True)
        
        yonehot = np.zeros_like(z2)
        yonehot[ytrue, np.arange(n)] = 1

        error2 = softmax - yonehot
        dw2 = (error2 @ self.a.T) / n               
        db2 = np.sum(error2, axis=1, keepdims=True) / n

        # HIDDEN GRADIENT
        error1 = (self.w2.T @ error2) * self.drelu(self.z)
        dw = (error1 @ self.x.T) / n
        db = np.sum(error1, axis=1, keepdims=True) / n

        return loss, {
            "dw": dw,
            "db": db,
            "dw2": dw2,
            "db2": db2,
        }

    def apply_gradients(self, gradients, alpha):
        self.w2 -= alpha * gradients["dw2"]
        self.b2 -= alpha * gradients["db2"]
        self.w -= alpha * gradients["dw"]
        self.b -= alpha * gradients["db"]

    def forward_after_step(self, x, gradients, alpha):
        x_t = x.T
        w = self.w - alpha * gradients["dw"]
        b = self.b - alpha * gradients["db"]
        w2 = self.w2 - alpha * gradients["dw2"]
        b2 = self.b2 - alpha * gradients["db2"]

        z = (w @ x_t) + b
        a = self.relu(z)
        return (w2 @ a) + b2

    def line_search_alpha(self, x, ytrue, gradients, alpha_candidates, current_loss):
        best_alpha = 0.0
        best_loss = current_loss

        for alpha in alpha_candidates:
            candidate_z2 = self.forward_after_step(x, gradients, alpha)
            candidate_loss = self.cross_entropy_loss(candidate_z2, ytrue)
            if np.isfinite(candidate_loss) and candidate_loss < best_loss:
                best_alpha = alpha
                best_loss = candidate_loss

        return best_alpha, best_loss

    def train_step(self, x, ytrue, alpha_candidates):
        z2 = self.forward(x)
        current_loss, gradients = self.gradients(z2, ytrue)
        alpha, loss = self.line_search_alpha(
            x,
            ytrue,
            gradients,
            alpha_candidates,
            current_loss,
        )
        self.apply_gradients(gradients, alpha)
        return loss, alpha

    def load(self, checkpoint_path):
        with np.load(checkpoint_path) as checkpoint:
            self.w = checkpoint["w"]
            self.b = checkpoint["b"]
            self.w2 = checkpoint["w2"]
            self.b2 = checkpoint["b2"]

    def save(self, checkpoint_path, epoch, loss, alpha=None):
        checkpoint = {
            "w": self.w,
            "b": self.b,
            "w2": self.w2,
            "b2": self.b2,
            "epoch": epoch,
            "loss": loss,
        }
        if alpha is not None:
            checkpoint["alpha"] = alpha

        np.savez(checkpoint_path, **checkpoint)


def checkpoint_epoch(checkpoint_path):
    try:
        with np.load(checkpoint_path) as checkpoint:
            return int(checkpoint["epoch"])
    except (OSError, KeyError, ValueError):
        return -1


def find_latest_checkpoint(save_dir):
    latest_path = None
    latest_epoch = 0

    if not os.path.isdir(save_dir):
        return latest_path, latest_epoch

    for filename in os.listdir(save_dir):
        if not filename.startswith("ann_epoch_") or not filename.endswith(".npz"):
            continue

        checkpoint_path = os.path.join(save_dir, filename)
        current_epoch = checkpoint_epoch(checkpoint_path)
        if current_epoch > latest_epoch:
            latest_path = checkpoint_path
            latest_epoch = current_epoch

    return latest_path, latest_epoch

############# MAIN #############

images = []
labels = []

archive_path = '/Users/macvee/Downloads/archive'
if os.path.exists(archive_path):
    for i in range(10):
        classdir = os.path.join(archive_path, str(i))
        if not os.path.isdir(classdir):
            continue

        for j in os.listdir(classdir):
            if j.startswith('.'):
                continue
                
            imgpath = os.path.join(classdir, j)
            try:
                img = Image.open(imgpath).convert("L")
                images.append(np.array(img))
                labels.append(i)
            except Exception:
                continue

if len(images) == 0:
    print("No images found! Creating dummy data for testing...")
    xflat = np.random.rand(100, 12600).astype("float32")
    y = np.random.randint(0, 10, size=(100,))
else:
    x = np.array(images)
    y = np.array(labels)
    xflat = x.reshape(x.shape[0], -1).astype("float32") / 255.0

ann = ANN()

epochs = 2000
save_every = 40
save_dir = "trained_data"
alpha_candidates = [0.0001, 0.0003, 0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1.0]
os.makedirs(save_dir, exist_ok=True)

checkpoint_path, start_epoch = find_latest_checkpoint(save_dir)
if checkpoint_path is None:
    print("No trained data checkpoint found. Starting from scratch.")
else:
    ann.load(checkpoint_path)
    print(f"Resumed trained data from {checkpoint_path} at epoch {start_epoch}.")

if start_epoch >= epochs:
    print(f"Training already complete: checkpoint epoch {start_epoch} >= target epochs {epochs}.")

for epoch in range(start_epoch, epochs):
    loss, alpha = ann.train_step(xflat, y, alpha_candidates)
    completed_epoch = epoch + 1
    
    if completed_epoch % 20 == 0 or completed_epoch == epochs:
        print(f"EPOCH: {completed_epoch} | LOSS: {loss:.4f} | ALPHA: {alpha:g}")

    if completed_epoch % save_every == 0 or completed_epoch == epochs:
        save_path = os.path.join(save_dir, f"ann_epoch_{completed_epoch:03d}.npz")
        ann.save(save_path, completed_epoch, loss, alpha)
        print(f"Saved trained data to {save_path}")


print("FINAL WEIGHTS (HIDDEN): ", ann.w.shape)  
print("FINAL BIASES (HIDDEN): ", ann.b.shape)     
print("FINAL WEIGHTS (OUT): ", ann.w2.shape)      
print("FINAL BIASES (OUT): ", ann.b2.shape)      
