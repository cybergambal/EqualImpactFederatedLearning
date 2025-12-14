import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import numpy as np
import time
import pandas as pd
import argparse
import os
from datetime import datetime
import sys
import random as rnd
import matplotlib.pyplot as plt
from collections import Counter
import random as rnd
from sklearn.metrics import confusion_matrix, f1_score
from FL_setting_NeurIPS_batuFlavor import FederatedLearning
import torch.nn.functional as F
from torchvision.models import resnet18, ResNet18_Weights
from byzfl import DataDistributor

# Start time
start_time = time.time()
# Simulate command-line arguments
sys.argv = [
    'placeholder_script_name',
    '--learning_rate', '0.01',
    '--epochs', '1',
    '--batch_size', '400',
    '--num_users', '100',
    '--fraction', '1',
    '--transmission_probability', '0.1',
    '--num_slots', '100',
    '--num_timeframes', '10000',
    '--user_data_size', '500',
    '--seeds', '56', #'3', #, '29', '85', '65',
    '--gamma_momentum', '0.15',
    '--use_memory_matrix', 'false',
    '--arrival_rate', '0.1',
    '--phase', '10', # number of timeframes per phase, there are in total five phases
    '--num_runs', '1',
    '--slotted_aloha', 'false', # for the NeurIPS paper, we don't consider random access channel
    '--num_memory_cells', '1',
    '--selected_mode', 'async_asymp_EI',
    '--cos_similarity', '4',
    '--cycle', '3',
    '--train_mode', 'all',
    '--keepProbAvail', '0.25',
    '--keepProbNotAvail', '0.9',
    '--bufferLimit', '1',
    '--theta_inner', '0.1',
    '--dirichlet_alpha', '0.5'
]

# Command-line arguments
parser = argparse.ArgumentParser(description="Federated Learning with Slotted ALOHA and CIFAR-10 Dataset")
parser.add_argument('--learning_rate', type=float, default=0.0001, help='Learning rate for training')
parser.add_argument('--epochs', type=int, default=3, help='Number of epochs for training')
parser.add_argument('--batch_size', type=int, default=128, help='Batch size for training')
parser.add_argument('--num_users', type=int, default=10, help='Number of users in federated learning')
parser.add_argument('--fraction', type=float, nargs='+', default=[0.1], help='Fraction for top-k sparsification')
parser.add_argument('--transmission_probability', type=float, default=0.1, help='Transmission probability for Slotted ALOHA')
parser.add_argument('--num_slots', type=int, default=10, help='Number of slots for Slotted ALOHA simulation')
parser.add_argument('--num_timeframes', type=int, default=15, help='Number of timeframes for simulation')
parser.add_argument('--seeds', type=int, nargs='+', default=[85, 12, 29], help='Random seeds for averaging results')
parser.add_argument('--gamma_momentum', type=float, nargs='+', default=[0.6], help='Momentum for memory matrix')
parser.add_argument('--use_memory_matrix', type=str, default='true', help='Switch to use memory matrix (true/false)')
parser.add_argument('--user_data_size', type=int, default=2000, help='Number of samples each user gets')
parser.add_argument('--arrival_rate', type=float, default=0.5,help='Arrival rate of new information')
parser.add_argument('--phase', type=int, default=5,help='When concept drift happens, when distribution change from one Class to another')
parser.add_argument('--num_runs', type=int, default=5,help='Number of simulations')
parser.add_argument('--slotted_aloha', type=str, default='true',help='Whether we use Slotted aloha in the simulation')
parser.add_argument('--num_memory_cells', type=int, default=6,help='Number of memory cells per client')
parser.add_argument('--selected_mode', type=str, default='async_Inner',help='Which setting we are using: genie_aided, vanilla, user_selection_cos, user_selection_cos_dis, user_selection_acc, user_selection_acc_increment, user_selection_aog, user_selection_norm')
parser.add_argument('--cos_similarity', type=int, default=2,help='What type of cosine similarity we want to test: cos2 = 2, cos4 = 4, ...')
parser.add_argument('--cycle', type=int, default=1,help='Number of cycles')
parser.add_argument('--train_mode', type=str, default='all',help='Which part of network we are training: all, dense, conv')
parser.add_argument('--keepProbAvail', type=float, default=1,help='Probability of a user keeping its state when available')
parser.add_argument('--keepProbNotAvail', type=float, default=1,help='Probability of a user keeping its state when not available')
parser.add_argument('--bufferLimit', type=int, default=1,help='Buffer size limit for how many users to wait before aggregation')
parser.add_argument('--theta_inner', type=float, default=0.9,help='Theta coeffcient for inner product test')
parser.add_argument('--dirichlet_alpha', type=float, default=0.5,help='Alpha coeffcient for dirichlet distribution')

args = parser.parse_args()

# Parsed arguments
learning_rate = args.learning_rate
epochs = args.epochs
batch_size = args.batch_size
num_users = args.num_users
fraction = args.fraction
transmission_probability = args.transmission_probability
num_slots = args.num_slots
num_timeframes = args.num_timeframes
seeds_for_avg = args.seeds
gamma_momentum = args.gamma_momentum
use_memory_matrix = args.use_memory_matrix.lower() == 'true'
user_data_size = args.user_data_size
tx_prob = args.transmission_probability
arrival_rate = args.arrival_rate
phase = args.phase
num_runs = args.num_runs
slotted_aloha = args.slotted_aloha
num_memory_cells = args.num_memory_cells
selected_mode = args.selected_mode
cycle = args.cycle
train_mode = args.train_mode
cos_similarity = args.cos_similarity
keepProbAvail = args.keepProbAvail
keepProbNotAvail = args.keepProbNotAvail
bufferLimit = args.bufferLimit
theta_inner = args.theta_inner
dirichlet_alpha = args.dirichlet_alpha

# Device configuration
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
torch.backends.cuda.matmul.allow_tf32 = True
print(f"\n{'*' * 50}\n*** Using device: {device} ***\n{'*' * 50}\n")

class_mappings = {
    0: [9, 6],
    1: [0, 5],
    2: [4, 1],
    3: [8, 7],
    4: [3, 5]
}

# Function to map original labels to new classes
def map_to_new_classes(original_labels):
    new_labels = np.zeros_like(original_labels)
    for new_class, original_classes in class_mappings.items():
        for original_class in original_classes:
            new_labels[original_labels == original_class] = new_class
    return new_labels

data_mode = "CIFAR"

if data_mode == "MNIST":
    # MNIST dataset and preprocessing
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    trainset = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    testset = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transform)
else:
    # CIFAR-10 dataset and preprocessing
    transform1 = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomCrop(32, 4),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],  # ImageNet stats
                         std=[0.229, 0.224, 0.225])
    ])
     
    transform2 = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],  # ImageNet stats
                         std=[0.229, 0.224, 0.225])
    ])
    trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform1)
    trainset.targets = torch.Tensor(trainset.targets).long()
    testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform2)


data_loader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=24)
params = {
    "data_distribution_name": "iid",
    #"distribution_parameter": dirichlet_alpha,
    "nb_honest": num_users,
    "data_loader": data_loader,
    "batch_size": batch_size,
}
distributor = DataDistributor(params)
TrainSetUsers = distributor.split_data()
testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=24)
print(len(trainset), "training samples loaded.")

assert len(trainset) >= num_users * user_data_size, "Dataset too small for requested user allocation!"

if data_mode == "MNIST":
    # CustomCNN Model
    class Model(nn.Module):
        def __init__(self, num_classes=10, train_mode=train_mode):
            """
            train_mode: 
                'all'    → train everything
                'dense'  → train only fc1 and fc2
                'conv'   → train only conv1 and conv2
            """
            super().__init__()
            self.conv1 = nn.Conv2d(1, 32, 3, padding=1)
            self.pool1 = nn.MaxPool2d(2, 2)
            self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
            self.pool2 = nn.MaxPool2d(2, 2)
            self.fc1 = nn.Linear(64 * 7 * 7, 128)
            self.fc2 = nn.Linear(128, num_classes)

            # Set requires_grad according to training mode
            if train_mode == 'dense':
                for param in self.conv1.parameters(): param.requires_grad = False
                for param in self.conv2.parameters(): param.requires_grad = False
            elif train_mode == 'conv':
                for param in self.fc1.parameters(): param.requires_grad = False
                for param in self.fc2.parameters(): param.requires_grad = False
            # 'all' means train everything → no changes needed

        def forward(self, x):
            x = self.pool1(F.relu(self.conv1(x)))
            x = self.pool2(F.relu(self.conv2(x)))
            x = x.view(x.size(0), -1)
            x = F.relu(self.fc1(x))
            return self.fc2(x)

else:
    class ResidualBlock(nn.Module):
        def __init__(self, inchannel, outchannel, stride=1):
            super(ResidualBlock, self).__init__() 
            self.dropout = nn.Dropout()
            self.left = nn.Sequential(
                nn.Conv2d(inchannel, outchannel, kernel_size=3, stride=stride, padding=1, bias=False),
                nn.GroupNorm(32,outchannel),
                nn.ReLU(inplace=True),
                nn.Conv2d(outchannel, outchannel, kernel_size=3, stride=1, padding=1, bias=False), 
                nn.GroupNorm(32,outchannel),
            )
            self.shortcut = nn.Sequential()
            if stride != 1 or inchannel != outchannel:
                self.shortcut = nn.Sequential(
                    nn.Conv2d(inchannel, outchannel, kernel_size=1, stride=stride, bias=False), 
                    nn.GroupNorm(32,outchannel),
                )
                
        def forward(self, x):
            out = self.left(x)
            out = out + self.shortcut(x)
            #out = self.dropout(out)
            out = F.relu(out)
            
            return out

    class ResNet(nn.Module):
        def __init__(self, ResidualBlock, num_classes=10):
            super(ResNet, self).__init__()
            self.inchannel = 64
            self.conv1 = nn.Sequential(
                nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False),
                nn.ReLU()
            )
            self.layer1 = self.make_layer(ResidualBlock, 64, 2, stride=1)
            self.layer2 = self.make_layer(ResidualBlock, 128, 2, stride=2)
            self.layer3 = self.make_layer(ResidualBlock, 256, 2, stride=2)        
            self.layer4 = self.make_layer(ResidualBlock, 512, 2, stride=2)        
            self.fc = nn.Linear(512, num_classes)
            self.dropout = nn.Dropout()
            
        def make_layer(self, block, channels, num_blocks, stride):
            strides = [stride] + [1] * (num_blocks - 1)
            layers = []
            for stride in strides:
                layers.append(block(self.inchannel, channels, stride))
                self.inchannel = channels
            return nn.Sequential(*layers)
        
        def forward(self, x):
            out = self.conv1(x)
            out = self.layer1(out)
            out = self.layer2(out)
            out = self.layer3(out)
            out = self.layer4(out)
            out = F.avg_pool2d(out, 4)
            out = out.view(out.size(0), -1)
            out = self.dropout(out)
            out = self.fc(out)
            return out    
    def Model(num_classes):
        return ResNet(ResidualBlock, num_classes)
# Partitioning User Data into Memory Cells
    
def evaluate_per_label_accuracy(model, testloader, device, num_classes=10):
    """
    Evaluate per-label accuracy on CIFAR-10 (original 10 labels, no remapping).

    Args:
        model (nn.Module): Trained model.
        testloader (DataLoader): DataLoader for the test dataset.
        device (torch.device): Device to run evaluation on.
        num_classes (int): Number of classes (default: 10 for CIFAR-10).

    Returns:
        dict: Per-label accuracy {label_index: accuracy_percentage}.
    """
    model.eval()
    with torch.no_grad():
        class_counts = {i: 0 for i in range(num_classes)}
        class_correct = {i: 0 for i in range(num_classes)}

        for images, labels in testloader:
            images, labels = images.to(device), labels.to(device)

            outputs = model(images)
            predictions = outputs.argmax(dim=1)

            for class_idx in range(num_classes):
                class_mask = (labels == class_idx)
                class_counts[class_idx] += class_mask.sum().item()
                class_correct[class_idx] += (predictions[class_mask] == class_idx).sum().item()
 
        total_samples = sum(class_counts.values())
        total_correct = sum(class_correct.values())
        
        per_label_accuracy = {}
        for class_idx in range(num_classes):
            if class_counts[class_idx] > 0:
                per_label_accuracy[class_idx] = 100 * class_correct[class_idx] / class_counts[class_idx]
            else:
                per_label_accuracy[class_idx] = 0.0

            print(f"Accuracy for Label {class_idx}: {per_label_accuracy[class_idx]:.2f}%")

        overall_accuracy = 100 * total_correct / total_samples if total_samples > 0 else 0
    
    return per_label_accuracy, overall_accuracy

def evaluate_user_model_accuracy(model, testloader, device) -> float:
    """
    Evaluate the accuracy of a user's model on the shared test set.

    Args:
        model (nn.Module): The trained model for the user.
        testloader (DataLoader): The test dataset loader (shared by all users).
        device (torch.device): The device to run evaluation on.

    Returns:
        float: Accuracy of the model on the test set.
    """
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in testloader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            preds = outputs.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)


    return correct / total if total > 0 else 0.0

testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False)

# Main training loop
seed_count = 1

for run in range(num_runs):
    rnd.seed(run)
    np.random.seed(run)
    torch.manual_seed(run)
    print(f"************ Run {run + 1} ************")

    for seed_index, seed in enumerate(seeds_for_avg):
        print(f"************ Seed {seed_count} ************")
        seed_count += 1
        # Define number of classes based on the dataset
        num_classes = 10  # CIFAR-10 has 10 classes

        # Initialize the model
       
        model = Model(num_classes=num_classes).to(device)
        
        
        criterion = nn.CrossEntropyLoss()
        #optimizer = optim.Adam(model.parameters(), lr=learning_rate)
        
        optimizer = optim.SGD(model.parameters(), momentum=0.9, lr=learning_rate, weight_decay=1e-4)

        #Initialize FL system once and for all for this seed.
        
        keepProbAvail = np.concatenate([
            np.full(num_users // 2, 0.1),  # First half: 0.1
            np.full(num_users - num_users // 2, 0.9)  # Second half: 0.9
        ])
        keepProbNotAvail = np.concatenate([
            np.full(num_users // 2, 0.9),  # First half: 0.9
            np.full(num_users - num_users // 2, 0.1)  # Second half: 0.1
        ])
        fl_system = FederatedLearning(
            selected_mode, num_users, device,
            cos_similarity, model, TrainSetUsers, epochs, optimizer, criterion, fraction,
            testloader, learning_rate, train_mode, keepProbAvail, keepProbNotAvail, bufferLimit, theta_inner
            )

        for timeframe in range(num_timeframes):
            print(f"******** Timeframe {timeframe + 1} ********")
#-----------------------------------------------------------------------------------------------------------------------------------------------------------
            # Run the FL mode and get updated weights
            new_weights = fl_system.run(run, seed_index, timeframe)
    
#-----------------------------------------------------------------------------------------------------------------------------------------------------------
            if (timeframe%(num_users//5) == 0): 
                # Updating the global model with the new aggregated weights 
                with torch.no_grad():
                    for param, saved in zip(model.parameters(), new_weights):
                        param.copy_(saved) 

            
                per_label_accuracy, accuracy = evaluate_per_label_accuracy(model, testloader, device, num_classes=10)


            # Store results and check if this is the best accuracy so far
            accuracy_distributions[run][seed_index][timeframe] = accuracy
            accuracy_per_labels[run][seed_index][timeframe] = per_label_accuracy

            correctly_received_packets_stats[run][seed_index][timeframe]['mean'] = fl_system.lp_cos_val
            correctly_received_packets_stats[run][seed_index][timeframe]['variance'] = 0

            torch.cuda.empty_cache()

            print(f"Mean Accuracy at Timeframe {timeframe + 1}: {accuracy:.2f}%")
        
        contribution = fl_system.contribution
        num_send = fl_system.num_send
        print("Number of successful users:", num_send/num_timeframes)
        print(contribution)
        del model
        del new_weights
        del fl_system
        torch.cuda.empty_cache()

# Prepare data for saving
current_time = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
save_dir = f"./results10slot1mem_{current_time}"
os.makedirs(save_dir, exist_ok=True)

# Save final results
final_results = []
for run in range(num_runs):
    for seed_index, seed in enumerate(seeds_for_avg):
        for timeframe in range(num_timeframes):
            final_results.append({
                'Run': run,
                'Seed': seed,
                'Timeframe': timeframe + 1,
                'Accuracy': accuracy_distributions[run][seed_index][timeframe],
                'Global Gradient Magnitude': global_grad_mag[run, seed_index, timeframe],
                'Packets Received': correctly_received_packets_stats[run][seed_index][timeframe]['mean'],
                'Variance Packets': correctly_received_packets_stats[run][seed_index][timeframe]['variance']
            })

            # Add additional per-timeframe statistics, independent of num_active_users
            final_results.append({
                'Run': run,
                'Seed': seed,
                'Timeframe': timeframe + 1,
                'Best Global Grad Mag': global_grad_mag[run, seed_index, timeframe],
                'Local Grad Mag': loc_grad_mag[run, seed_index, timeframe].tolist(),
                'Local Grad Mag with Memory': loc_grad_mag_memory[run, seed_index, timeframe].tolist(),
                'Memory Matrix Magnitude': memory_matrix_mag[run, seed_index, timeframe].tolist(),
                'Best Accuracy': accuracy_distributions[run][seed_index][timeframe],
                'Best-Successful Users': contribution.tolist()[timeframe%(num_timeframes//num_users)] 
            })


final_results_df = pd.DataFrame(final_results)
file_path = os.path.join(save_dir, 'final_results.csv')
final_results_df.to_csv(file_path, index=False)
print(f"Final results saved to: {file_path}")

# Save correctly received packets statistics to CSV
end_time = time.time()
elapsed_time = end_time - start_time

# Save run summary
summary_content = (
    f"Start Time: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(start_time))}\n"
    f"End Time: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(end_time))}\n"
    f"Elapsed Time: {elapsed_time:.2f} seconds\n"
    f"Arguments: {vars(args)}\n"
)

summary_file_path = os.path.join(save_dir, 'run_summary.txt')
with open(summary_file_path, 'w') as summary_file:
    summary_file.write(summary_content)

print(f"Run summary saved to: {summary_file_path}")
