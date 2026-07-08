import os
import numpy as np
from PIL import Image

# WRITING SHALLOW ANN FROM SCRATCH (AI DEBUGGED)

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
    
    def backward(self, z2, ytrue, alpha=0.1):
        n = z2.shape[1]                            
        
        shifted = z2 - np.max(z2, axis=0, keepdims=True)
        softmax = np.exp(shifted) / np.sum(np.exp(shifted), axis=0, keepdims=True)
        
        probs = np.clip(softmax[ytrue, np.arange(n)], 1e-15, 1.0)
        celoss = -np.log(probs)
        
        yonehot = np.zeros_like(z2)
        yonehot[ytrue, np.arange(n)] = 1

        error2 = softmax - yonehot
        dw2 = (error2 @ self.a.T) / n               
        db2 = np.sum(error2, axis=1, keepdims=True) / n

        # HIDDEN GRADIENT
        error1 = (self.w2.T @ error2) * self.drelu(self.z)
        dw = (error1 @ self.x.T) / n
        db = np.sum(error1, axis=1, keepdims=True) / n

        # UPDATE
        self.w2 -= alpha * dw2
        self.b2 -= alpha * db2
        self.w -= alpha * dw
        self.b -= alpha * db

        return np.mean(celoss)

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

epochs = 1000
save_every = 40
save_dir = "trained_data"
os.makedirs(save_dir, exist_ok=True)

for epoch in range(epochs):
    forward_output = ann.forward(xflat)
    loss = ann.backward(forward_output, y, alpha=0.1)
    completed_epoch = epoch + 1
    
    if epoch % 20 == 0 or epoch == epochs - 1:
        print(f"EPOCH: {epoch} | LOSS: {loss:.4f}")

    if completed_epoch % save_every == 0:
        save_path = os.path.join(save_dir, f"ann_epoch_{completed_epoch:03d}.npz")
        np.savez(
            save_path,
            w=ann.w,
            b=ann.b,
            w2=ann.w2,
            b2=ann.b2,
            epoch=completed_epoch,
            loss=loss,
        )
        print(f"Saved trained data to {save_path}")


print("FINAL WEIGHTS (HIDDEN): ", ann.w.shape)  
print("FINAL BIASES (HIDDEN): ", ann.b.shape)     
print("FINAL WEIGHTS (OUT): ", ann.w2.shape)      
print("FINAL BIASES (OUT): ", ann.b2.shape)      
