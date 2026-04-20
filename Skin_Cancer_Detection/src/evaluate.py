import torch
import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report, f1_score, recall_score, accuracy_score, precision_score, fbeta_score
from torch.utils.data import DataLoader
from torchvision import transforms


current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.append(current_dir)


try:
    from model_def import HybridSkinModel
    from dataset import HAM10000
    from utils import get_data_splits
except ImportError:
    from .model_def import HybridSkinModel
    from .dataset import HAM10000
    from .utils import get_data_splits

def save_metrics_table(metrics_dict, results_dir):
    """Generates and saves a professional metrics table image."""
    fig, ax = plt.subplots(figsize=(6, 3))
    ax.axis('off')
    
    table_data = [[k, f"{v:.3f}"] for k, v in metrics_dict.items()]
    column_labels = ["Metric", "Score"]
    
    table = ax.table(cellText=table_data, colLabels=column_labels, 
                     cellLoc='center', loc='center', colColours=["#0056b3", "#0056b3"])
    
  
    table.auto_set_font_size(False)
    table.set_fontsize(12)
    table.scale(1.2, 2.5)
    
 
    for (row, col), cell in table.get_celld().items():
        if row == 0:
            cell.get_text().set_color('white')
            cell.get_text().set_weight('bold')

    plt.title("Overall Performance of the Proposed Model", pad=20, fontweight='bold')
    plt.savefig(os.path.join(results_dir, "performance_metrics_table.png"), bbox_inches='tight', dpi=300)
    plt.close()

def evaluate():
  
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    METADATA_PATH = os.path.join(BASE_DIR, "Data", "HAM10000_metadata.csv")
    IMG_DIR = os.path.join(BASE_DIR, "Data", "all_images")
    MODEL_PATH = os.path.join(BASE_DIR, "model", "hybrid_model.pth")
    RESULTS_DIR = os.path.join(BASE_DIR, "results")
    
    os.makedirs(RESULTS_DIR, exist_ok=True)

    CLASS_NAMES = ['Nevi', 'Melanoma', 'Benign Keratosis', 'Basal Cell', 'Actinic', 'Vascular', 'Dermatofibroma']

 
    print(" Loading data for evaluation...")
    _, val_df = get_data_splits(METADATA_PATH)
    val_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    val_ds = HAM10000(val_df, IMG_DIR, transform=val_transform)
    val_loader = DataLoader(val_ds, batch_size=32, shuffle=False)


    if not os.path.exists(MODEL_PATH):
        print(f" Error: Model not found at {MODEL_PATH}.")
        return

    model = HybridSkinModel(num_classes=7).to(DEVICE)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.eval()

    all_preds, all_labels = [], []

    print(f" Evaluating model on {DEVICE}...")
    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            outputs = model(images)
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

 
    total_accuracy = accuracy_score(all_labels, all_preds)
    precision = precision_score(all_labels, all_preds, average='macro')
    recall = recall_score(all_labels, all_preds, average='macro')
    f1 = f1_score(all_labels, all_preds, average='macro')
    f05 = fbeta_score(all_labels, all_preds, beta=0.5, average='macro')

    report = classification_report(all_labels, all_preds, target_names=CLASS_NAMES)
    
    print("\n" + "="*30)
    print(f"✅ TOTAL ACCURACY: {total_accuracy*100:.2f}%")
    print("="*30)
    print("\n--- Detailed Classification Report ---")
    print(report)

   
    metrics_to_save = {
        "Precision": precision,
        "Recall": recall,
        "F1-score": f1,
        "F0.5-score": f05
    }
    save_metrics_table(metrics_to_save, RESULTS_DIR)

    with open(os.path.join(RESULTS_DIR, "metrics_report.txt"), "w") as f:
        f.write(f"TOTAL OVERALL ACCURACY: {total_accuracy*100:.2f}%\n")
        f.write(f"Macro Precision: {precision:.4f}\n")
        f.write(f"Macro Recall: {recall:.4f}\n")
        f.write(f"Macro F1-score: {f1:.4f}\n")
        f.write(f"Macro F0.5-score: {f05:.4f}\n")
        f.write("-" * 30 + "\n")
        f.write(report)

 
    cm = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES)
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title(f'Confusion Matrix (Overall Acc: {total_accuracy*100:.1f}%)')
    plt.savefig(os.path.join(RESULTS_DIR, "confusion_matrix.png"))
    plt.close()

    metrics_names = ['Total Accuracy', 'F1-Score (Macro)', 'Recall (Macro)']
    values = [total_accuracy, f1, recall]

    plt.figure(figsize=(8, 6))
    bars = plt.bar(metrics_names, values, color=['#4CAF50', '#2196F3', '#FF9800'])
    plt.ylim(0, 1.1)
    plt.ylabel('Score (0.0 - 1.0)')
    plt.title('Hybrid Model: Final Performance Summary')
    
    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2, yval + 0.02, f'{yval*100:.1f}%', ha='center', fontweight='bold')
    
    plt.savefig(os.path.join(RESULTS_DIR, "overall_metrics.png"))
    plt.close()

    print(f"✅ Evaluation Finished! Results saved in: {RESULTS_DIR}")

if __name__ == "__main__":
    evaluate()