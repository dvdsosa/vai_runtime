# metrics.py
import torch
from torcheval.metrics.functional import (
    multiclass_accuracy,
    multiclass_precision,
    multiclass_recall,
    multiclass_f1_score
)

def compute_metrics(ground_truth_classes_vector, inferred_classes_vector, num_classes):
    """Computes the accuracy (micro), precision, recall and f1-metrics (the last three using macro)."""
    #override if needed
    #num_classes = 2 

    ground_truth_classes_vector = torch.tensor(ground_truth_classes_vector).long()
    inferred_classes_vector = torch.tensor(inferred_classes_vector).long()

    # calcular métricas mediante torcheval
    acc = multiclass_accuracy(inferred_classes_vector, ground_truth_classes_vector, average="micro", num_classes=num_classes, k=1)
    prec = multiclass_precision(inferred_classes_vector, ground_truth_classes_vector, average="macro", num_classes=num_classes)
    rec = multiclass_recall(inferred_classes_vector, ground_truth_classes_vector, average="macro", num_classes=num_classes)
    f1_score = multiclass_f1_score(inferred_classes_vector, ground_truth_classes_vector, average="macro", num_classes=num_classes)

    return acc.item(), prec.item(), rec.item(), f1_score.item()
