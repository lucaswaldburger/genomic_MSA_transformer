import argparse
import math

import numpy as np
import torch
from matplotlib import pyplot as plt
from sklearn import metrics
from sklearn.metrics import confusion_matrix
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataloader import OperonLoader
from operon_transformer import operon_transformer_classifier


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate the operon classifier on validation and test sets")
    parser.add_argument("--subsample-length", type=int, default=300)
    parser.add_argument("--depth", type=int, default=11)
    parser.add_argument("--batch-size", type=int, default=60)
    parser.add_argument("--device", type=str, default="cuda:3")
    parser.add_argument("--img-set", type=str, default="bsubtilis", choices=["bsubtilis", "ecoli"])
    parser.add_argument("--max-num-genes", type=int, default=13)
    parser.add_argument("--include-position", action="store_true")
    parser.add_argument("--include-sequence", action="store_true")
    parser.add_argument("--data-csv", type=str, default="data.csv")
    parser.add_argument("--transformer-path", type=str, default="saved_models/operon_transformer_stable_last.pt")
    parser.add_argument("--model-path", type=str, default="saved_models/stable_operon_transformer_classifier_lr.001_last.pt")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main():
    args = parse_args()

    torch.random.manual_seed(args.seed)

    val_data = OperonLoader(args.data_csv,
                            subsample_length=args.subsample_length,
                            include_position=args.include_position,
                            include_sequence=args.include_sequence,
                            max_num_genes=args.max_num_genes,
                            split_key="valid",
                            im_set=args.img_set,
                            device=args.device)

    test_data = OperonLoader(args.data_csv,
                             subsample_length=args.subsample_length,
                             include_position=args.include_position,
                             include_sequence=args.include_sequence,
                             max_num_genes=args.max_num_genes,
                             split_key="test",
                             im_set=args.img_set,
                             device=args.device)

    val_dataloader = DataLoader(val_data, batch_size=args.batch_size)
    test_dataloader = DataLoader(test_data, batch_size=args.batch_size)

    model = operon_transformer_classifier(
        depth=args.depth,
        ckpt_path=args.transformer_path,
        max_num_genes=args.max_num_genes,
        alignment_length=args.subsample_length,
        include_position=args.include_position,
        include_sequence=args.include_sequence,
        stable=False,
        attn_dropout=0,
        ff_dropout=0).to(args.device)

    model.load_state_dict(torch.load(args.model_path))
    model.eval()

    preds = []
    labels = []

    with torch.no_grad():
        for batch in tqdm(val_dataloader):
            msa, score, label = batch
            out = model(msa, score)
            if len(preds) == 0:
                preds = out[0]
                labels = label.T[0]
            else:
                preds = torch.cat((preds, out[0]), dim=0)
                labels = torch.cat((labels, label.T[0]), dim=0)

    fpr, tpr, thresholds = metrics.roc_curve(labels.cpu(), preds.cpu())

    auc = metrics.roc_auc_score(labels.cpu(), preds.cpu())
    with plt.style.context('ggplot'):
        plt.figure(dpi=150, clear=True, edgecolor='white', facecolor='white', frameon=False, tight_layout=True)
        plt.title(f'ROC Curve for Validation Set\nAUC={str(auc)}', color='black')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.plot(fpr, tpr)
        plt.savefig('roc_curve.png')
        plt.show()

    cutoff = thresholds[np.argmax(tpr - fpr)]

    preds_test = []
    labels_test = []

    with torch.no_grad():
        for batch in tqdm(test_dataloader):
            msa, score, label = batch
            out = model(msa, score)
            if len(preds_test) == 0:
                preds_test = out[0] >= cutoff
                labels_test = label.T[0]
            else:
                preds_test = torch.cat((preds_test, out[0] >= cutoff), dim=0)
                labels_test = torch.cat((labels_test, label.T[0]), dim=0)

    tn, fp, fn, tp = confusion_matrix(labels_test.long().cpu(), preds_test.long().cpu(), labels=[0, 1]).ravel()

    sensitivity = tp / (tp + fn)
    precision = tp / (tp + fp)
    specificity = tn / (tn + fp)
    accuracy = (tp + tn) / (tp + tn + fp + fn)
    mcc = (tp * tn - fp * fn) / (math.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)))
    f1 = 2 * tp / (2 * tp + fp + fn)

    print(f'Sensitivity: {sensitivity}')
    print(f'Precision: {precision}')
    print(f'Specificity: {specificity}')
    print(f'Accuracy: {accuracy}')
    print(f'MCC: {mcc}')
    print(f'F1: {f1}')
    print(f'Cutoff: {cutoff}')
    print(f'Test set size: {len(test_data)}')


if __name__ == "__main__":
    main()
