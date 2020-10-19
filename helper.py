import torch
import torch.nn as nn
import torch.nn.functional as F
import pickle
import numpy as np

from features import exp_features as features

def parse_champs(game):
    with open('data/champ_dict.pkl', 'rb') as f:
        champ_dict = pickle.load(f)
        
    picked = []
    picked.append([champ_dict[champ] for champ in game])
    picked = picked[0]
    
    five_hot1 = np.zeros((len(champ_dict),), dtype=int)
    for j in picked[0:5]:
        five_hot1[j] = 1
    five_hot2 = np.zeros((len(champ_dict),), dtype=int)
    for k in picked[5:10]:
        five_hot2[k] = 1
    return np.concatenate((five_hot1, five_hot2))

def load_model(path):
    model = nn.Sequential(*features)
    model.load_state_dict(torch.load(f'models/{path}'))
    model.eval()
    return model

def custom_test(game,model='quarter_finals.pth'):
    '''
    Returns a torch tensor that predicts the game output
    
    '''
    net = load_model(model)
    ten_hot = parse_champs(game)
    custom_X = torch.tensor(ten_hot, dtype=torch.float)
    
    net.eval()
    with torch.no_grad():
        output = net(custom_X)
    
    return F.softmax(output, dim=-1)

def get_unique():
    '''
    Returns a list of the unique champions available in the model 
    if a champion is not available the model will not work
    '''
    with open('data/champ_dict.pkl', 'rb') as f:
        champ_dict = pickle.load(f)
    return champ_dict.keys()