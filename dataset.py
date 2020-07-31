import torch
import numpy as np
import os
from tqdm import tqdm
import cv2

def load_characters(root, alphabet):
    '''
    Loads the characters from a given alphabet
    Args:
        root (str): Root directory
        alphabet (str): folder name of alphabet
    Returns:
        (tuple) of:
            (list): images
            (list): labels
    '''
    X = []
    y = []
    
    alphabet_path = os.path.join(root, alphabet)
    characters = os.listdir(alphabet_path)
    
    for char in characters:
        char_path = os.path.join(alphabet_path, char)
        images = os.listdir(char_path)
        
        for img in images:
            image = cv2.imread(os.path.join(char_path, img))
            image = cv2.resize(image, (28, 28)) / 255
            
            X.append(image)
            y.append(f'{alphabet}_{char}')
    
    return X, y


def load_data(root):
    '''
    Loads the full omniglot dataset from a root directory
    Args:
        root (str): path of omniglot dataset
    Returns:
        (tuple) of:
            (ndarray): images
            (ndarray): labels
    '''
    X_data = []
    y_data = []
    
    print('Loading Data')
    
    alphabets = os.listdir(root)
    for alphabet in tqdm(alphabets):
        X, y = load_characters(root, alphabet)
        X_data.extend(X)
        y_data.extend(y)
    
    return np.array(X_data), np.array(y_data)

def extract_sample(X_data, y_data, task_params):
    '''
    Extract a random sample as a k-shot n-way task
    Args:
        X_data (ndarray): images
        y_data (ndarray): labels
        task_params (dict): task parameters dictionary containing k_shot, n_way and n_query
    Returns:
        (tuple): of train and test samples 
    '''
    k_shot = task_params['k_shot']
    n_way = task_params['n_way']
    n_query = task_params['n_query']
    
    X_train = []
    y_train = []
    
    X_test = []
    y_test = []
    
    # Randomly select n_way classes
    sampled_cls = np.random.choice(np.unique(y_data), n_way, replace=False)
        
    for i, c in enumerate(sampled_cls):
        # Select images belonging to that class
        X_data_c = X_data[y_data == c]
        
        # Sample k_shot+n_query images
        sample_images = np.random.permutation(X_data_c)[:(k_shot+n_query)]
        
        # Add to lists
        X_train.extend(sample_images[:k_shot])
        X_test.extend(sample_images[k_shot:])
        
        y_train.extend([i] * k_shot)
        y_test.extend([i] * n_query)
    
    # Shuffle indices
    train_idx = np.random.permutation(len(X_train))
    test_idx = np.random.permutation(len(X_test))
    
    # Convert to tensor and permute the images as channels first and use the shuffle indices
    X_train = torch.Tensor(X_train).float().permute(0, 3, 1, 2)[train_idx]
    y_train = torch.Tensor(y_train)[train_idx].long()
    
    X_test = torch.Tensor(X_test).float().permute(0, 3, 1, 2)[test_idx]
    y_test = torch.Tensor(y_test)[test_idx].long()
      
    return (X_train, y_train), (X_test, y_test)