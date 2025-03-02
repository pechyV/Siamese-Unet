import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from modules.dataset import ChangeDetectionDataset
from model.siamese_unet import get_model
from modules.utils import visualize_results, setup_logging, save_checkpoint, load_checkpoint, save_final_model, load_pretrained_model
from modules.early_stop import EarlyStopping
import logging
import os

# Nastavení logování
setup_logging()

def train(load_pretrain, model, train_dataloader, val_dataloader, criterion, optimizer, device, num_epochs, checkpoint_dir="./checkpoints/", patience = 3):

    start_epoch = 0 # set checkpoint > 0
    checkpoint_path = os.path.join(checkpoint_dir, f"checkpoint_epoch_{start_epoch}.pth")
    
    # Pokud existuje checkpoint, načteme jej
    if not load_pretrain and start_epoch > 0 and os.path.exists(checkpoint_path):
        start_epoch = load_checkpoint(model, optimizer, checkpoint_path)
        model.to(device)
        logging.info(f"Pokračování tréninku od epochy {start_epoch+1}")
    
    early_stopping = EarlyStopping(patience)

    for epoch in range(start_epoch, num_epochs):
        model.train()  # Přepnutí modelu do režimu trénování
        epoch_loss = 0.0
        for t1, t2, mask in train_dataloader:
            t1, t2, mask = t1.to(device), t2.to(device), mask.to(device)
            optimizer.zero_grad()
            outputs = model(t1, t2)
            loss = criterion(outputs, mask)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

        logging.info(f"Epoch {epoch+1}/{num_epochs}, Training Loss: {epoch_loss/len(train_dataloader):.4f}")

        # Validace po každé epoše
        model.eval()  # Přepnutí modelu do režimu evaluace
        val_loss = 0.0
        with torch.no_grad():
            for t1, t2, mask in val_dataloader:
                t1, t2, mask = t1.to(device), t2.to(device), mask.to(device)
                outputs = model(t1, t2)
                loss = criterion(outputs, mask)
                val_loss += loss.item()

        logging.info(f"Epoch {epoch+1}/{num_epochs}, Validation Loss: {val_loss/len(val_dataloader):.4f}")

        # Zavolání EarlyStopping
        early_stopping(val_loss / len(val_dataloader))  # Předání průměrné validační ztráty

        if early_stopping.early_stop:
            logging.info("EARLY STOPPING: trénování zastaveno.")
            break

        # Uložení checkpointu po každé epoše
        save_checkpoint(model, optimizer, epoch, checkpoint_dir)
        
        # Vizualizace výsledků po každé epoše
        with torch.no_grad():
            t1, t2, mask = t1[0], t2[0], mask[0]  # První obrázek v dávce
            visualize_results(t1, t2, mask, (outputs > 0.5).float()[0], epoch+1)

if __name__ == "__main__":
    """ Parametry """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = get_model(0)
    load_pretrain = False
    learning_rate = 0.001
    num_epochs = 40
    batch_size = 8
    patience = 15
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), learning_rate)
    train_root_dir = "./test_dataset/train/"
    val_root_dir = "./test_dataset/val/"
    out_model = "./trained_model/siamese_unet.pth"
    pretrained_model = "./trained_model/siamese_unet.pth"
    checkpoint_dir = "./checkpoints"
    
    transform = transforms.Compose([
        transforms.ToTensor()
    ])
    
    train_dataset = ChangeDetectionDataset(train_root_dir, transform=transform)
    val_dataset = ChangeDetectionDataset(val_root_dir, transform=transform)

    train_dataloader = DataLoader(train_dataset, batch_size, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size, shuffle=False)

    # Načtení předtrénovaného modelu
    if load_pretrain:           
        if load_pretrained_model(model, pretrained_model):
            model.to(device)

    train(load_pretrain, model, train_dataloader, val_dataloader, criterion, optimizer, device, num_epochs, checkpoint_dir, patience)
    
    # Uložení modelu po trénování
    save_final_model(model, out_model)