import torch.nn as nn

from cnn_simple import CnnSimple
from train_common import build_arg_parser, run_training


def build_model(num_classes: int) -> nn.Module:
    return CnnSimple(num_classes=num_classes)


def main():
    args = build_arg_parser(
        default_batch_size=32, default_lr=1e-3, default_patience=50,
    ).parse_args()
    run_training(
        build_model=build_model,
        dataset_mode="cnn5",
        wandb_extra={"model": "cnn_simple", "input": "1x68x136"},
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
