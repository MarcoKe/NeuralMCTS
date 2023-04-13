from collections import Counter
from typing import List
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.stats import entropy


def calculate_entropy_from_operations_list(operation_types: List) -> float:
    """Calculates the entropy of a list of operations."""
    counts = Counter(operation_types)
    probabilities = [count/len(operation_types) for count in counts.values()]
    entropy_val = entropy(probabilities)
    return entropy_val

class ProbabilityAdjustmentNetwork(nn.Module):
    """A simple neural network that takes in a vector of probabilities and adjusts them to have a certain entropy."""
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(hidden_size, output_size)
        self.fc2.bias.data.fill_(-torch.log(torch.tensor(1.0/output_size)))

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return F.softmax(x, dim=0)


def get_entropy_loss(pred, target):
    """Calculates the loss between the entropy of the predicted probabilities and the target entropy."""
    distribution = torch.distributions.Categorical(probs=pred)
    entropy = distribution.entropy()    
    loss = 0.5*(entropy-target)**2
    return loss


def get_rounded_entropy_loss(pred, target, output_size):
    """Calculates rounded entropy loss. This is used to ensure that the probabilities are not too small."""
    pred = pred[pred>round(1/output_size,4)]
    pred = torch.cat((pred, torch.Tensor([1- torch.sum(pred)])), 0)
    distribution = torch.distributions.Categorical(probs=pred)
    entropy = distribution.entropy()    
    loss = 0.5*(entropy-target)**2
    return loss

class EntropyOptimizer:
    """A class that optimizes the probabilities of a list of operations to have a certain entropy."""
    def __init__(self, output_size: int, hidden_size: int=20, learning_rate: float=0.1, num_epochs: int=200, max_episodes: int=200, precision: float=0.1):
        self.input_size = output_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.learning_rate = learning_rate
        self.num_epochs = num_epochs
        self.max_episodes = max_episodes
        self.precision = precision
        
    def _find_probabilities(self, target, low_entropy=False):
        """Finds the probabilities that will result in the target entropy."""
        network = ProbabilityAdjustmentNetwork(input_size=self.input_size, hidden_size=self.hidden_size, output_size=self.output_size)
        optimizer = torch.optim.AdamW(network.parameters(), lr=self.learning_rate)
        input_data = torch.ones(self.input_size)

        
        # Train the network
        for epoch in range(self.num_epochs):
            # Forward pass
            output = network(input_data)
            if low_entropy:
                loss = get_entropy_loss(output, target)
            else:
                loss = get_rounded_entropy_loss(output, target, self.output_size)
                
            if loss <= 0.0005:
                break

            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        with torch.no_grad():
            output = network(input_data).detach()

        return output
    
    def _entropy_distribution_finder(self, entropy_ratio):
        """
        Finds the probabilities that will result in the target entropy.
        It will initialize new network for the entropy search until the target entropy is reached.
        """
        max_entropy = calculate_entropy_from_operations_list(list(range(self.output_size)))
        target_entropy = max_entropy*entropy_ratio
        i = 0
        abs_entropy = target_entropy
        best_output = []
        current_best = target_entropy
        
        while abs_entropy >= self.precision:
            if i > self.max_episodes:
                print(f"failure with size={self.output_size}, and entropy_ratio={entropy_ratio}, best_output={entropy(best_output)-target_entropy}")
                break
            # I realized that when entropy is high, it helps network to remove very low ratios from the distribution list.
            # I remove the ratios that are smaller than 1/size of the distribution (these are the ones that wouldn't be present in the distribution anyway).
            if entropy_ratio < 0.5:
                output = self._find_probabilities(target=target_entropy, low_entropy=True)
            else:
                output = self._find_probabilities(target=target_entropy, low_entropy=False)
            
            # Here I again remove the ones that are smaller then the output size and add the rest to the last one, just to make sure they sum up to 1.
            filtered_output = [round(float(i), 4) for i in output if i > 1/self.output_size]
            filtered_output.append(round(1-sum(filtered_output), 4))
            
            abs_entropy = abs(entropy(filtered_output)-target_entropy)
            if abs_entropy <= current_best:
                current_best=abs_entropy
                best_output=filtered_output
            i += 1
            
        print(f"found entropy = {entropy(best_output)} vs target={target_entropy} at epoch={i}. found/max = {entropy(best_output)/max_entropy}. Output={best_output}")
        return best_output
    
    def find_entropies(self):
        entropies_dict = {}
        for i in range(2, 9):
            ratio = round(i*0.1, 2)
            outputs = self._entropy_distribution_finder(entropy_ratio=ratio)
            entropies_dict[ratio] = outputs
        return entropies_dict
