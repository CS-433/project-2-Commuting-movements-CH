import torch
import torch.nn as nn
import tqdm

from sklearn.metrics import f1_score, roc_auc_score
import numpy as np

INPUT_DIM = 29

class FCNN(nn.Module):
    def __init__(self, layer_nodes = [100]):
        super().__init__()

        hidden_layers = []
        for i in range(1, len(layer_nodes)):
            hidden_layers.append(nn.Linear(layer_nodes[i-1], layer_nodes[i]))
            hidden_layers.append(nn.ReLU())
        

        self.stack = nn.Sequential(
            nn.Linear(INPUT_DIM, layer_nodes[0]),
            nn.ReLU(),
            *hidden_layers,
            nn.Linear(layer_nodes[-1], 1),
        )

    def forward(self, x):
        logits = self.stack(x)
        return logits
    
def train_model(x_train, y_train, x_val, y_val, num_epochs, pos_weight, batch_size, patience, lr, layer_nodes, weight_decay, **kwargs):

    
    pos_weight = torch.tensor(pos_weight).to("cpu")      

    train_set = torch.utils.data.TensorDataset(x_train, y_train)
    val_set = torch.utils.data.TensorDataset(x_val, y_val)
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_set, batch_size=batch_size, shuffle=False)
    
    model = FCNN(layer_nodes)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight) # does NOT need sigmoid layer in model!
    optimizer = torch.optim.Adam(params=model.parameters(), lr=lr, weight_decay=weight_decay)


    train_losses = []
    val_losses = []
    val_rocs = []
    val_f1s = []
    
    best_model_state = None

    # early stopping if roc doesn't change. Not f1 because we might be able to optimise that later using a different threshold
    best_val_roc = 0 # will be beaten in first epoch
    patience_left = patience
    val_roc_epsilon = 1e-3
    
    
    for epoch in tqdm.tqdm(range(num_epochs), desc=f'Training {num_epochs} epochs:'):
        model.train()
        
        running_loss = 0.0
        for x, y in train_loader:
            optimizer.zero_grad()
            outputs = model(x).squeeze(1) # shape (batch_size, 1) -> shape (batch_size, )

            loss = criterion(outputs, y)
            
            running_loss += loss.item()
            loss.backward()        

            optimizer.step()
        avg_loss = running_loss / len(train_loader) # slightly off if last batch is smaller.
        
        # Evaluation

        model.eval()
        running_loss = 0.0
        val_preds = np.array([]) 
                
        with torch.no_grad():
            for x, y in val_loader:    
                logits = model(x).squeeze(1)
                loss = criterion(logits, y)
                        
                running_loss += loss.item()
                
                val_preds = np.append(val_preds, torch.sigmoid(logits).numpy())
                
        avg_val_loss = running_loss / len(val_loader)
        val_roc = roc_auc_score(y_val, val_preds)
        val_f1 = f1_score(y_val, val_preds)

        train_losses.append(round(avg_loss, 5))
        val_losses.append(round(avg_val_loss, 5))
        val_rocs.append(round(val_roc, 5))
        val_f1s.append(round(val_f1, 5))

        if val_roc > best_val_roc + val_roc_epsilon:
            best_val_roc = val_roc
            best_model_state = model.state_dict().copy()
            patience_left = patience
        else:
            patience_left -= 1
        if patience_left <= 0:
            print("Out of patience. Ending training")
            break
        
    return best_model_state, (train_losses, val_losses, val_rocs, val_f1s)
