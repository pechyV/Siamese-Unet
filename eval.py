import torch
import torch.utils.data
import torchvision.transforms as transforms
from sklearn.metrics import confusion_matrix
from tqdm import tqdm
from modules.dataset import ChangeDetectionDataset
from model.siamese_unet import SiameseUNet
import numpy as np
from modules.utils import setup_logging
import logging

# Nastavení logování
setup_logging()

def evaluate(root_dir="./test_dataset/test/"):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Načtení modelu
    model_path = "./trained_model/siamese_unet.pth"
    model = SiameseUNet().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    # Načtení testovacího datasetu
    test_transform = transforms.Compose([
        transforms.ToTensor()
    ])

    test_dataset = ChangeDetectionDataset(root_dir, transform=test_transform)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=16, shuffle=False, num_workers=0)

    # Inicializace confusion matrix a metrik
    c_matrix = {'tn': 0, 'fp': 0, 'fn': 0, 'tp': 0}
    total_iou = 0
    total_pixel_acc = 0
    num_samples = 0

    with torch.no_grad():
        tbar = tqdm(test_loader)
        for t1, t2, labels in tbar:
            t1, t2, labels = t1.to(device), t2.to(device), labels.to(device)

            # Inferenční predikce
            outputs = model(t1, t2)
            outputs = outputs.squeeze(1)  # Odstranění nadbytečné dimenze
            preds = (outputs > 0.5).long()  # Binarizace výstupu

            # Konverze do NumPy
            labels_np = labels.cpu().numpy().flatten()
            preds_np = preds.cpu().numpy().flatten()

            # Ošetření případu, kdy dataset obsahuje jen jednu třídu
            unique_labels = np.unique(labels_np)
            if len(unique_labels) == 1:
                cmat = confusion_matrix(labels_np, preds_np, labels=[0, 1])
            else:
                cmat = confusion_matrix(labels_np, preds_np)

            # Ujistíme se, že confusion_matrix vrací 4 hodnoty
            if cmat.shape == (2, 2):
                tn, fp, fn, tp = cmat.ravel()
            else:
                tn, fp, fn, tp = 0, 0, 0, 0

            # Sčítání hodnot
            c_matrix['tn'] += tn
            c_matrix['fp'] += fp
            c_matrix['fn'] += fn
            c_matrix['tp'] += tp

            # Výpočet IoU pro aktuální batch
            intersection = tp
            union = tp + fp + fn
            iou = intersection / union if union > 0 else 0
            total_iou += iou

            # Výpočet pixel accuracy
            pixel_acc = (tp + tn) / (tp + tn + fp + fn)
            total_pixel_acc += pixel_acc
            num_samples += 1

    # Výpočet metrik přesnosti
    tn, fp, fn, tp = c_matrix['tn'], c_matrix['fp'], c_matrix['fn'], c_matrix['tp']
    P = tp / (tp + fp) if (tp + fp) > 0 else 0
    R = tp / (tp + fn) if (tp + fn) > 0 else 0
    F1 = 2 * P * R / (R + P) if (R + P) > 0 else 0
    mean_iou = total_iou / num_samples if num_samples > 0 else 0
    mean_pixel_acc = total_pixel_acc / num_samples if num_samples > 0 else 0

    # Výpis metrik
    logging.info("Evaluace:")
    logging.info(f'Precision: {P:.4f}')
    logging.info(f'Recall: {R:.4f}')
    logging.info(f'F1-Score: {F1:.4f}')
    logging.info(f'IoU: {mean_iou:.4f}')
    logging.info(f'Pixel Accuracy: {mean_pixel_acc:.4f}')

if __name__ == '__main__':
    evaluate()