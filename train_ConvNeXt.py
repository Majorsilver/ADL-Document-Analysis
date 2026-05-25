import torch.nn as nn
from torchvision.models import convnext_tiny, ConvNeXt_Tiny_Weights

from train_common import build_arg_parser, run_training
from convnext import build_model


def main():
    args = build_arg_parser(
        default_batch_size=128, default_lr=1e-5, default_patience=50,
    ).parse_args()
    run_training(
        build_model=build_model,
        dataset_mode="convnext",
        wandb_extra={"model": "convnext_tiny", "weights": "IMAGENET1K_V1"},
        task=args.task,
        batch_size=args.batch_size,
        lr=args.lr,
        max_epochs=args.max_epochs,
        patience=args.patience,
        num_workers=args.num_workers,
        out_dir=args.out_dir,
        run_name=args.run_name,
        augment=args.augment,
        resume=args.resume,
        use_wandb=not args.no_wandb,
    )


if __name__ == "__main__":
    main()
