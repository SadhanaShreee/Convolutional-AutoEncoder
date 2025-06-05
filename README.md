# Convolutional Autoencoder for Image Denoising

## AIM

To develop a convolutional autoencoder for image denoising application.

## Problem Statement and Dataset

Image data is often corrupted by noise in real-world applications such as OCR, medical imaging, and photography. Autoencoders can learn to reconstruct clean images from noisy inputs. In this project, we use the MNIST dataset and add Gaussian noise to simulate corrupted images.

## DESIGN STEPS

### STEP 1:

Import necessary libraries such as torch, torchvision, and matplotlib.

### STEP 2:

Define a function to add Gaussian noise to images.

### STEP 3:

Define the architecture for the convolutional autoencoder using nn.Sequential.

### STEP 4:

Use the MSE Loss function and the Adam optimizer.

### STEP 5:

Train the model on noisy images and validate it against clean labels.

### STEP 6:

Visualize original, noisy, and denoised outputs for a qualitative understanding.

## PROGRAM

### Name: JWALAMUKHI S

### Register Number: 212223040079

```
class DenoisingAutoencoder(nn.Module):
    def __init__(self):
        super(DenoisingAutoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1),  # 28x28 -> 28x28
            nn.ReLU(),
            nn.MaxPool2d(2, 2),             # 28x28 -> 14x14
            nn.Conv2d(16, 8, 3, padding=1), # 14x14 -> 14x14
            nn.ReLU(),
            nn.MaxPool2d(2, 2)              # 14x14 -> 7x7
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(8, 16, 2, stride=2),    # 7x7 -> 14x14
            nn.ReLU(),
            nn.ConvTranspose2d(16, 8, 2, stride=2),    # 14x14 -> 28x28
            nn.ReLU(),
            nn.Conv2d(8, 1, 3, padding=1),             # 28x28
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x

model = DenoisingAutoencoder().to(device)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training Function
def train(model, loader, criterion, optimizer, epochs=5):
    model.train()
    for epoch in range(epochs):
        running_loss = 0.0
        for images, _ in loader:
            images = images.to(device)
            noisy_images = add_noise(images).to(device)

            outputs = model(noisy_images)
            loss = criterion(outputs, images)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
        print(f"Epoch [{epoch+1}/{epochs}], Loss: {running_loss/len(loader):.4f}")

def visualize_denoising(model, loader, num_images=10):
    model.eval()
    with torch.no_grad():
        for images, _ in loader:
            images = images.to(device)
            noisy_images = add_noise(images).to(device)
            outputs = model(noisy_images)
            break

    images = images.cpu().numpy()
    noisy_images = noisy_images.cpu().numpy()
    outputs = outputs.cpu().numpy()

   
    plt.figure(figsize=(18, 6))
    for i in range(num_images):
        # Original
        ax = plt.subplot(3, num_images, i + 1)
        plt.imshow(images[i].squeeze(), cmap='gray')
        ax.set_title("Original")
        plt.axis("off")

        # Noisy
        ax = plt.subplot(3, num_images, i + 1 + num_images)
        plt.imshow(noisy_images[i].squeeze(), cmap='gray')
        ax.set_title("Noisy")
        plt.axis("off")

        # Denoised
        ax = plt.subplot(3, num_images, i + 1 + 2 * num_images)
        plt.imshow(outputs[i].squeeze(), cmap='gray')
        ax.set_title("Denoised")
        plt.axis("off")

    plt.tight_layout()
    plt.show()
```

## OUTPUT

### Model Summary

![image](https://github.com/user-attachments/assets/e2f2a35f-1867-47b1-85fb-8ab897c14f36)

### Original vs Noisy Vs Reconstructed Image


![6](https://github.com/user-attachments/assets/d70afd4d-fddb-4c44-b71a-37cbd3604019)


## RESULT

A convolutional autoencoder was successfully implemented using PyTorch. The model was able to reconstruct clean images from noisy inputs, demonstrating its ability to learn meaningful image features for denoising.
