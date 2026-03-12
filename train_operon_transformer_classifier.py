import argparse
import math

import numpy as np
import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataloader import OperonLoader
from operon_transformer import operon_transformer_classifier


def parse_args():
    parser = argparse.ArgumentParser(description="Train the operon classifier on pretrained transformer embeddings")
    parser.add_argument("--subsample-length", type=int, default=300)
    parser.add_argument("--depth", type=int, default=11)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--device", type=str, default="cuda:3")
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--img-set", type=str, default="bsubtilis", choices=["bsubtilis", "ecoli"])
    parser.add_argument("--max-num-genes", type=int, default=13)
    parser.add_argument("--include-position", action="store_true")
    parser.add_argument("--include-sequence", action="store_true")
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--data-csv", type=str, default="data.csv")
    parser.add_argument("--save-dir", type=str, default="saved_models")
    parser.add_argument("--ckpt-path", type=str, default="saved_models/operon_transformer_stable_last.pt")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main():
    args = parse_args()

    torch.random.manual_seed(args.seed)

    train_data = OperonLoader(args.data_csv,
                              subsample_length=args.subsample_length,
                              subsampling_method="regular_random",
                              shuffle_rows=False,
                              flip_orientation=True,
                              include_position=args.include_position,
                              include_sequence=args.include_sequence,
                              max_num_genes=args.max_num_genes,
                              split_key="train",
                              im_set=args.img_set,
                              device=args.device)
    val_data = OperonLoader(args.data_csv,
                            subsample_length=args.subsample_length,
                            include_position=args.include_position,
                            include_sequence=args.include_sequence,
                            max_num_genes=args.max_num_genes,
                            split_key="valid",
                            im_set=args.img_set,
                            device=args.device)

    train_dataloader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True)
    val_dataloader = DataLoader(val_data, batch_size=args.batch_size)

    model = operon_transformer_classifier(
        depth=args.depth,
        ckpt_path=args.ckpt_path,
        max_num_genes=args.max_num_genes,
        alignment_length=args.subsample_length,
        include_position=args.include_position,
        include_sequence=args.include_sequence,
        stable=False,
        attn_dropout=.1,
        ff_dropout=.1).to(args.device)

    optimizer = AdamW(model.parameters(),
                      lr=args.lr,
                      betas=(0.9, 0.96),
                      weight_decay=4.5e-2,
                      amsgrad=True)

    epoch_train_losses = []
    epoch_val_losses = []
    best_val_loss = float("inf")

    torch.cuda.empty_cache()

    for epoch in range(args.epochs):
        train_losses = []
        val_losses = []
        model.train()
        for j, batch in tqdm(enumerate(train_dataloader)):
            msa, score, label = batch
            optimizer.zero_grad()
            out = model(msa, score)
            loss = torch.nn.MSELoss()(out, label)
            loss.backward()

            if j % 500 == 0:
                print(f'{j}: Mean Loss: {np.mean([v for v in train_losses if not math.isnan(v) and not math.isinf(v)])}')

            optimizer.step()
            train_losses.append(loss.item())

        model.eval()
        with torch.no_grad():
            for batch in tqdm(val_dataloader):
                msa, score, label = batch
                out = model(msa, score)
                loss = torch.nn.MSELoss()(out, label)
                val_losses.append(loss.item())

        train_losses = [v for v in train_losses if not math.isnan(v) and not math.isinf(v)]
        val_losses = [v for v in val_losses if not math.isnan(v) and not math.isinf(v)]
        epoch_train_losses.append(np.sum(train_losses) / len(train_dataloader))
        epoch_val_losses.append(np.sum(val_losses) / len(val_dataloader))

        print(f'Epoch {epoch} - Train Loss: {epoch_train_losses[-1]:.4f} - Val Loss: {epoch_val_losses[-1]:.4f}')

        if epoch_val_losses[-1] < best_val_loss:
            best_val_loss = epoch_val_losses[-1]
            torch.save(
                model.state_dict(),
                f'{args.save_dir}/stable_operon_transformer_classifier_lr.001_{args.img_set}_{epoch}.pt')

        torch.save(
            model.state_dict(),
            f'{args.save_dir}/stable_operon_transformer_classifier_lr.001_last.pt')

        with open(f"{args.save_dir}/stable_classifier_lr.001_{args.img_set}.txt", "w") as outfile:
            outfile.write('Train: ' + ",".join(str(v) for v in epoch_train_losses) + '\n')
            outfile.write('Val: ' + ",".join(str(v) for v in epoch_val_losses))


if __name__ == "__main__":
    main()
