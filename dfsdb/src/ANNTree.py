import torch
from torch import nn, optim
from torch.utils.data import DataLoader, random_split
from torchvision import transforms
import numpy as np
import logging
from doodleLoaderSimple import DoodleDatasetSimple
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt
from PIL import Image

# Setting up log file
logging.basicConfig(filename='error.log', level=logging.ERROR, format='%(asctime)s - %(levelname)s: %(message)s')

# Number to class labels mapping for tree image classifier
class_dict = {
    0: 'depression',
    1: 'introvert',
    2: 'ambitious'
}

# Loading the data from the .csv file
try:
    data = np.genfromtxt('../data/treeData.csv', dtype=int, delimiter=',', names=True)
except FileNotFoundError as e:
    logging.error(f"File not found error: {e}")
    exit(1)
except Exception as e:
    logging.error(f"An unexpected error occurred while loading data: {e}")
    exit(1)

# Match the image IDs to the ID values in the .csv file
translation_dict = dict(
    zip([f'{id}.png' for id in data['id']], data['class']))

# Prepare each image to be passed as a Tensor product to the model
data_transforms = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),  # If images are RGB, convert to grayscale
    transforms.Resize((28, 28)),  # Resize to a fixed size
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

# Prepare the data by matching it to its label and transforming it to a Tensor product
try:
    treedata = DoodleDatasetSimple('../images/tree/', data_transforms, translation_dict)
except Exception as e:
    logging.error(f"An error occurred while preparing data: {e}")
    exit(1)

# Split the data into training and validation sets
train_len = int(len(treedata) * 0.8)
test_len = len(treedata) - train_len
train_set, val_set = random_split(treedata, [train_len, test_len])

# Load the data in batches
train_loader = DataLoader(train_set, batch_size=4, shuffle=True, num_workers=0, drop_last=True)
test_loader = DataLoader(val_set, batch_size=4, shuffle=False, num_workers=0, drop_last=True)

class ANNClassifier(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(ANNClassifier, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, num_classes)
    
    def forward(self, x):
        x = x.view(x.size(0), -1)  # Flatten the input
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        return out

# Set the device to use GPU if available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Define the ANN model
input_size = 28 * 28  # Input size after flattening the image
hidden_size = 128
num_classes = 3
model = ANNClassifier(input_size, hidden_size, num_classes).to(device)

# Define the criterion and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

def train_model(model, device, train_loader, criterion, optimizer, epochs=10):
    model.train()
    for epoch in range(epochs):
        for i, pictures in enumerate(train_loader):
            images = pictures['image'].to(device)
            labels = pictures['class'].to(device).long()

            # Forward pass
            outputs = model(images)
            loss = criterion(outputs, labels)

            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if (i + 1) % 100 == 0:
                print(f'Epoch [{epoch+1}/{epochs}], Step [{i+1}/{len(train_loader)}], Loss: {loss.item():.4f}')
    
    print("Finished Training")

def validate_model(model, device, test_loader):
    model.eval()
    with torch.no_grad():
        n_correct = 0
        n_samples = 0
        for pictures in test_loader:
            images = pictures['image'].to(device)
            labels = pictures['class'].to(device).long()
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            n_samples += labels.size(0)
            n_correct += (predicted == labels).sum().item()

        acc = 100.0 * n_correct / n_samples
        print(f'Accuracy of the model on the test images: {acc:.2f}%')

# Train and validate the model
try:
    train_model(model, device, train_loader, criterion, optimizer, epochs=10)
except Exception as e:
    logging.error(f"An error occurred during training: {e}")
    exit(1)

try:
    validate_model(model, device, test_loader)
except Exception as e:
    logging.error(f"An error occurred during validation: {e}")

# Define the function to load and preprocess a single image
def load_and_preprocess_image(image_path):
    # Load the image
    image = Image.open(image_path).convert('RGB')
    # Preprocess the image
    image = data_transforms(image).unsqueeze(0).to(device)
    return image

# Function to get predictions for a single image
def get_prediction(model, image_tensor):
    with torch.no_grad():
        model.eval()
        output = model(image_tensor)
        _, predicted = torch.max(output, 1)
        predicted_label = class_dict[predicted.item()]
    return predicted_label

# Load a fixed image
fixed_image_path = '../images/tree/300.png'
fixed_image = load_and_preprocess_image(fixed_image_path)

# Get the prediction for the fixed image
prediction = get_prediction(model, fixed_image)

# Display the prediction
print(f'Prediction for the fixed image: {prediction}')

# Generate classification report for the test set
def generate_classification_report(model, test_loader):
    y_true = []
    y_pred = []

    with torch.no_grad():
        for pictures in test_loader:
            images = pictures['image'].to(device)
            labels = pictures['class'].to(device).long()
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            y_true.extend(labels.cpu().numpy())
            y_pred.extend(predicted.cpu().numpy())

    report = classification_report(y_true, y_pred, target_names=list(class_dict.values()))
    print("Classification Report:")
    print(report)

# Call the function to generate classification report
generate_classification_report(model, test_loader)

# Initialize a dictionary to store class counts
class_counts = {0: 0, 1: 0, 2: 0}

# Iterate through the dataset and count occurrences of each class label
for data_item in treedata:
    class_label = data_item['class']
    class_counts[class_label] += 1

# Print the class counts
print("Class Counts for Tree Images:")
for label, count in class_counts.items():
    print(f"Class {label}: {count} occurrences")