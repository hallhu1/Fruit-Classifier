import kagglehub
from torchvision import transforms
from pathlib import Path

def classifier():
    # Download latest version
    dataset_root = Path(kagglehub.dataset_download("moltean/fruits"))
    
    # Using the fruits360 100x00 dataset, as it is the largest from kagglehub
    fruits_360_images = dataset_root.append("fruits-360_100x100/fruits-360")
    training_set = dataset_root.append("Training")
    evaluation_set = dataset_root.append("Test")

    Pretrained_image_size = 224

    train_tfms = transforms.Compose([
        transforms.RandomResizedCrop(Pretrained_image_size, scale=(0.6, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(0.2,0.2,0.2,0.1),
        transforms.RandomRotation(15),
        transforms.ToTensor(),
        transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225]),
    ])
    eval_tfms = transforms.Compose([
        transforms.Resize((Pretrained_image_size, Pretrained_image_size)),
        transforms.ToTensor(),
        transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225]),
    ])   
        

if __name__ == "__main__":
    exit(classifier())
