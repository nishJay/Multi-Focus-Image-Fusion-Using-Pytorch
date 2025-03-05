import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from sklearn.model_selection import train_test_split
from skimage.metrics import structural_similarity as ssim
import matplotlib.pyplot as plt

# ------------------------------
# Configuration and Paths
# ------------------------------

BASE_PATH = 'images/MFI-WHU'
SOURCE1_DIR = os.path.join(BASE_PATH, 'source_1')
SOURCE2_DIR = os.path.join(BASE_PATH, 'source_2')
FULL_CLEAR_DIR = os.path.join(BASE_PATH, 'full_clear')

BATCH_SIZE = 4
NUM_EPOCHS = 30
IMAGE_SIZE = (256, 256)
LEARNING_RATE = 1e-3
RANDOM_STATE = 42


# ------------------------------
# Dataset Definition
# ------------------------------
class FusionDataset(Dataset):
    """Custom dataset for image fusion tasks.
    
    Each sample consists of two source images (concatenated along the channel dimension)
    and a corresponding full clear ground truth image.
    """
    def __init__(self, image_pairs):
        """
        Args:
            image_pairs (list of tuples): Each tuple contains paths for (source1, source2, full_clear) images.
        """
        self.image_pairs = image_pairs
        self.transform = transforms.Compose([
            transforms.Resize(IMAGE_SIZE),
            transforms.ToTensor()
        ])

    def __len__(self):
        return len(self.image_pairs)

    def __getitem__(self, idx):
        source1_path, source2_path, full_clear_path = self.image_pairs[idx]
        source1 = self.transform(Image.open(source1_path))
        source2 = self.transform(Image.open(source2_path))
        full_clear = self.transform(Image.open(full_clear_path))
        # Concatenate the source images along the channel dimension (resulting in 6 channels)
        input_tensor = torch.cat((source1, source2), dim=0)
        return input_tensor, full_clear


# ------------------------------
# Model Definition
# ------------------------------
class FusionNet(nn.Module):
    """Convolutional neural network for image fusion."""
    def __init__(self):
        super(FusionNet, self).__init__()
        self.encoder = nn.Sequential(
            self.conv_block(6, 64),
            self.conv_block(64, 128),
            self.conv_block(128, 256)
        )
        self.decoder = nn.Sequential(
            self.conv_block(256, 128),
            self.conv_block(128, 64),
            nn.Conv2d(64, 3, kernel_size=3, padding=1),
            nn.Sigmoid()
        )

    def conv_block(self, in_channels, out_channels):
        """Helper function to create a convolutional block."""
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        features = self.encoder(x)
        return self.decoder(features)


# ------------------------------
# Utility Functions
# ------------------------------
def get_image_pairs(num_images=120):
    """Generate a list of image path tuples.
    
    Args:
        num_images (int): Total number of image sets (default is 120).
        
    Returns:
        List of tuples: Each tuple contains (source1_path, source2_path, full_clear_path)
    """
    image_pairs = []
    for i in range(1, num_images + 1):
        src1 = os.path.join(SOURCE1_DIR, f'{i}.jpg')
        src2 = os.path.join(SOURCE2_DIR, f'{i}.jpg')
        full = os.path.join(FULL_CLEAR_DIR, f'{i}.jpg')
        image_pairs.append((src1, src2, full))
    return image_pairs


def compare_images(source1, source2, fused, ground_truth, idx, phase="Train"):
    """Display source, fused, and ground truth images side by side.
    
    Args:
        source1, source2, fused, ground_truth (Tensor): Image tensors.
        idx (int): Index of the image.
        phase (str): Indicator for phase (Train/Test).
    """
    def tensor_to_img(tensor):
        img = tensor.cpu().numpy().transpose(1, 2, 0)
        return np.clip(img, 0, 1)
    
    imgs = [tensor_to_img(source1), tensor_to_img(source2),
            tensor_to_img(fused), tensor_to_img(ground_truth)]
    
    titles = [f'Source 1 Image {idx}', f'Source 2 Image {idx}',
              f'Fused Image {idx} ({phase})', f'Ground Truth Image {idx}']
    
    fig, axs = plt.subplots(1, 4, figsize=(20, 6))
    for ax, img, title in zip(axs, imgs, titles):
        ax.imshow(img)
        ax.set_title(title)
        ax.axis('off')
    plt.tight_layout()
    plt.show()


def evaluate_fusion(fused, ground_truth):
    """Compute the Structural Similarity Index (SSIM) between fused and ground truth images.
    
    Args:
        fused, ground_truth (Tensor): Image tensors.
        
    Returns:
        float: SSIM score.
    """
    fused_np = fused.cpu().numpy().transpose(1, 2, 0)
    gt_np = ground_truth.cpu().numpy().transpose(1, 2, 0)
    # Use multichannel SSIM for color images
    return ssim(fused_np, gt_np, multichannel=True)


def evaluate_ssim(model, loader, device, phase="Test"):
    """Evaluate the model on a dataset and display SSIM scores.
    
    Args:
        model (nn.Module): The fusion model.
        loader (DataLoader): DataLoader for the dataset.
        device (torch.device): Computation device.
        phase (str): 'Train' or 'Test' indicator.
        
    Returns:
        float: Average SSIM score.
    """
    model.eval()
    ssim_scores = []

    with torch.no_grad():
        for i, (inputs, targets) in enumerate(loader):
            inputs, targets = inputs.to(device), targets.to(device)
            # Split concatenated sources if needed (for display purposes)
            source1, source2 = torch.split(inputs, 3, dim=1)
            fused = model(inputs)
            score = evaluate_fusion(fused[0], targets[0])
            ssim_scores.append(score)
            print(f"{phase} image {i+1}, SSIM score: {score:.4f}")
            compare_images(source1[0], source2[0], fused[0], targets[0], i+1, phase=phase)

    avg_ssim = np.mean(ssim_scores)
    print(f"\nAverage SSIM on {phase} set: {avg_ssim:.4f}")
    return avg_ssim


def train_and_evaluate(model, train_loader, test_loader, device, num_epochs=NUM_EPOCHS):
    """Train the model and evaluate its performance on both training and test sets.
    
    Args:
        model (nn.Module): The fusion network.
        train_loader, test_loader (DataLoader): Data loaders.
        device (torch.device): Computation device.
        num_epochs (int): Number of training epochs.
    """
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    
    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0.0
        print(f"\nEpoch [{epoch+1}/{num_epochs}]")
        
        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        
        avg_loss = epoch_loss / len(train_loader)
        print(f"Average Training Loss: {avg_loss:.4f}")

    print("\n--- Evaluating on Training Set ---")
    train_ssim = evaluate_ssim(model, train_loader, device, phase="Train")

    print("\n--- Evaluating on Testing Set ---")
    test_ssim = evaluate_ssim(model, test_loader, device, phase="Test")
    
    return train_ssim, test_ssim


# ------------------------------
# Main Execution
# ------------------------------
def main():
    # Prepare image pairs and split dataset
    image_pairs = get_image_pairs(num_images=120)
    train_pairs, test_pairs = train_test_split(image_pairs, test_size=0.2, random_state=RANDOM_STATE)
    
    train_dataset = FusionDataset(train_pairs)
    test_dataset = FusionDataset(test_pairs)
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=1)
    
    # Set device and initialize model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = FusionNet().to(device)
    
    # Train model and evaluate performance
    train_ssim, test_ssim = train_and_evaluate(model, train_loader, test_loader, device, num_epochs=NUM_EPOCHS)
    print(f"\nFinal Training SSIM: {train_ssim:.4f}")
    print(f"Final Testing SSIM: {test_ssim:.4f}")


if __name__ == "__main__":
    main()
