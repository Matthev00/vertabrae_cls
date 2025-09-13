import argparse
from pathlib import Path

import numpy as np
import torch
import yaml
from sklearn.metrics import auc, roc_curve
import matplotlib.pyplot as plt

from src.config import LABELS_FILE_PATH, TENSOR_DIR
from src.data_utils.dataloader import get_dataloders
from src.modeling.model_factory import create_model
from src.utils import set_seed


def evaluate_model(model, dataloader, device):
    model.eval()
    all_probs = []
    all_labels = []
    
    with torch.no_grad():
        for batch in dataloader:
            inputs = batch['image'].to(device)
            labels = batch['label'].to(device)
            
            outputs = model(inputs)
            probs = torch.sigmoid(outputs).cpu().numpy()
            
            all_probs.extend(probs)
            all_labels.extend(labels.cpu().numpy())
    
    return np.array(all_probs), np.array(all_labels)


def plot_roc_curves(train_probs, train_labels, test_probs, test_labels, save_path):
    plt.style.use('seaborn-v0_8-paper')
    plt.figure(figsize=(10, 8))
    
    fpr_train, tpr_train, _ = roc_curve(train_labels, train_probs)
    roc_auc_train = auc(fpr_train, tpr_train)
    plt.plot(fpr_train, tpr_train, 'b-', linewidth=2, 
            label=f'Zbiór treningowy (AUC = {roc_auc_train:.3f})')
    
    fpr_test, tpr_test, _ = roc_curve(test_labels, test_probs)
    roc_auc_test = auc(fpr_test, tpr_test)
    plt.plot(fpr_test, tpr_test, 'r-', linewidth=2,
            label=f'Zbiór testowy (AUC = {roc_auc_test:.3f})')
    
    plt.plot([0, 1], [0, 1], 'k--', linewidth=1.5, alpha=0.7)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Odsetek fałszywie pozytywnych', fontsize=12)
    plt.ylabel('Odsetek prawdziwie pozytywnych', fontsize=12)
    plt.title('Krzywe ROC dla klasyfikacji binarnej', fontsize=14, pad=20)
    plt.legend(loc="lower right", fontsize=10, frameon=True)
    plt.grid(True, linestyle='--', alpha=0.7)
    
    plt.xticks(fontsize=10)
    plt.yticks(fontsize=10)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    return roc_auc_train, roc_auc_test


def main():
    parser = argparse.ArgumentParser(description='Evaluate binary classification model')
    parser.add_argument('--config', type=str, required=True, help='Path to model config YAML')
    parser.add_argument('--weights', type=str, required=True, help='Path to model weights')
    parser.add_argument('--output-dir', type=str, default='reports/figures', help='Directory to save plots')
    args = parser.parse_args()
    
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    set_seed(42)
    
    train_loader, test_loader = get_dataloders(
        labels_file_path=LABELS_FILE_PATH,
        tensor_dir=TENSOR_DIR,
        train_transforms=None,
        val_transforms=None,
        batch_size=config['batch_size'],
        num_workers=0,
        balance_train=config.get('balance_train', False),
        train_split=0.7,
        binary_class=True,
        balancer_type=config.get('balancer_type', 'base'),
    )
    
    model = create_model(
        model_type=config['model_type'],
        num_classes=1,
        model_depth=config.get('model_depth', 18),
        shortcut_type=config.get('shortcut_type', 'B'),
        freeze_backbone=config.get('freeze_backbone', False),
        device=device,
    )
    
    model.load_state_dict(torch.load(args.weights))
    model = model.to(device)
    
    train_probs, train_labels = evaluate_model(model, train_loader, device)
    test_probs, test_labels = evaluate_model(model, test_loader, device)
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    roc_plot_path = output_dir / 'roc.png'
    roc_auc_train, roc_auc_test = plot_roc_curves(
        train_probs, train_labels, 
        test_probs, test_labels,
        roc_plot_path
    )
    
    print(f'\nResults:')
    print(f'Training ROC AUC: {roc_auc_train:.3f}')
    print(f'Test ROC AUC: {roc_auc_test:.3f}')
    print(f'\nROC curves have been saved to: {roc_plot_path}')


if __name__ == '__main__':
    main()
