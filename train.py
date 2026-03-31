import os
import sys
import warnings
import yaml
import argparse
from datetime import datetime

import models
import models.losses
import models.metrics
import models.optimizers
import options
import data as datasets
from torch.utils.data import DataLoader
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning import loggers as pl_loggers


def load_config(config_path: str) -> dict:
    with open(config_path, 'r') as f:
        return yaml.safe_load(f) or {}


if __name__ == "__main__":

    pre = argparse.ArgumentParser(add_help=False)
    pre.add_argument('--config', type=str, default=None)
    pre_args, _ = pre.parse_known_args()

    config = {}
    if pre_args.config and os.path.exists(pre_args.config):
        config = load_config(pre_args.config)
        print(f"Loaded config: {pre_args.config}")

    parser = argparse.ArgumentParser()
    parser = options.set_argparse_defs(parser)
    parser = options.add_argparse_args(parser)

    parser.add_argument('--max_epochs',           type=int,  default=config.get('max_epochs',           100))
    parser.add_argument('--accelerator',          type=str,  default=config.get('accelerator',          'gpu'))
    parser.add_argument('--devices',              type=int,  default=config.get('devices',              1))
    parser.add_argument('--precision',                       default=config.get('precision',            '16-mixed'))
    parser.add_argument('--log_every_n_steps',    type=int,  default=config.get('log_every_n_steps',    10))
    parser.add_argument('--num_sanity_val_steps', type=int,  default=config.get('num_sanity_val_steps', 0))
    parser.add_argument('--deterministic',        action='store_true', default=config.get('deterministic', False))
    parser.add_argument('--config',               type=str,  default=None)

    config_defaults = {
        k: v for k, v in config.items()
        if k not in ('max_epochs','accelerator','devices','precision',
                     'log_every_n_steps','num_sanity_val_steps','deterministic')
        and v is not None
    }
    parser.set_defaults(**config_defaults)

    args = parser.parse_args()
    run_time = datetime.now().strftime('%Y%m%d_%H%M%S')
    start_epoch = 0
    if args.load and args.load != '':
        import re
        match = re.search(r'epoch[=_](\d+)', os.path.basename(args.load))
        if match:
            start_epoch = int(match.group(1))

    run_name = f"{args.remarks}_{run_time}_from_ep{start_epoch:03d}"

    args.default_root_dir = os.path.join('checkpoints', run_name)
    os.makedirs(args.default_root_dir, exist_ok=True)

    warnings.filterwarnings('ignore', "The \\`srun\\` command is available on your system but is not used.")
    warnings.filterwarnings('ignore', "torch.meshgrid: in an upcoming release, it will be required to pass the indexing argument")
    warnings.filterwarnings('ignore', "Detected call of \\`lr_scheduler.step\\(\\)\\` before \\`optimizer.step\\(\\)\\`")
    warnings.filterwarnings('ignore', "Checkpoint directory .* exists and is not empty")

    print(f"  run_name     : {run_name}")
    print(f"  checkpoints  : {args.default_root_dir}")
    print(f"  loss         : {args.loss}")
    print(f"  network      : {args.network}")
    print(f"  patch_size   : {args.patch_size}")
    print(f"  max_patches  : {args.max_patches}")
    print(f"  max_epochs   : {args.max_epochs}")
    print(f"  batch_size   : {args.batch_size}")

    seed_everything(args.seed, workers=True)

    loss = models.losses.__dict__[args.loss]

    # Dataset
    train_data, valid_data, tests_data = datasets.__dict__[args.dataset](
        dataroot            = args.dataroot,
        seed                = args.seed,
        fraction            = args.fraction,
        augment             = args.augment,
        patch_size          = args.patch_size,
        stride              = args.patch_stride,
        subsample_factors   = args.subsample_factors,
        input_type          = args.input_type,
        max_patches_per_vol = args.max_patches,
    )

    train_loader = DataLoader(train_data, batch_size=args.batch_size,       shuffle=True,  drop_last=True,  num_workers=4, pin_memory=True)
    valid_loader = DataLoader(valid_data, batch_size=args.valid_batch_size, shuffle=False, num_workers=4,   pin_memory=True)
    tests_loader = DataLoader(tests_data, batch_size=args.valid_batch_size, shuffle=False, num_workers=4,   pin_memory=True)

    network = models.__dict__[args.network](
        train_data.__numinput__(), train_data.__numclass__(),
        pretrained   = args.pretrained,
        padding      = args.padding,
        padding_mode = args.padding_mode,
        drop         = args.drop_rate,
        skip         = args.global_skip,
    )
    optim = models.optimizers.__dict__[args.optim](
        network.parameters(),
        lr           = args.lr_start,
        momentum     = args.momentum,
        weight_decay = args.decay,
        nesterov     = args.nesterov,
    )

    train_metrics = [models.metrics.__dict__[args.train_metrics[i]]() for i in range(len(args.train_metrics))]
    valid_metrics = [models.metrics.__dict__[args.valid_metrics[i]]() for i in range(len(args.valid_metrics))]
    tests_metrics = [models.metrics.__dict__[args.tests_metrics[i]]() for i in range(len(args.tests_metrics))]


    callbacks = [
        ModelCheckpoint(
            monitor        = args.monitor,
            mode           = args.monitor_mode,
            dirpath        = args.default_root_dir,
            filename       = 'best_epoch={epoch:03d}_valloss={val_loss:.4f}',
            save_top_k     = 1,
            save_last      = False,
            auto_insert_metric_name = False,
        ),

        ModelCheckpoint(
            monitor        = None,
            dirpath        = args.default_root_dir,
            filename       = 'latest_epoch={epoch:03d}',
            save_top_k     = 1,
            every_n_epochs = 1,
            save_last      = False,
            auto_insert_metric_name = False,
        ),

        ModelCheckpoint(
            monitor        = None,
            dirpath        = args.default_root_dir,
            filename       = 'epoch={epoch:03d}',
            save_top_k     = -1,         
            every_n_epochs = 10,
            save_last      = False,
            auto_insert_metric_name = False,
        ),
        models.ProgressBar(refresh_rate=5),
    ]

    logger = pl_loggers.TensorBoardLogger(
        save_dir   = 'logs',
        name       = args.remarks,
        version    = run_time,           
        default_hp_metric = False,
    )

    loader  = models.__dict__[args.trainee].load_from_checkpoint if args.load != '' else models.__dict__[args.trainee]
    checkpt = args.load if (args.load != '' and os.path.exists(args.load)) else None

    if args.load != '' and checkpt is None:
        print(f"WARNING: checkpoint '{args.load}' not found — starting fresh")

    trainee = loader(
        checkpoint_path = checkpt,
        model           = network,
        optimizer       = optim,
        train_data      = train_data,
        valid_data      = valid_data,
        tests_data      = tests_data,
        loss            = loss,
        train_metrics   = train_metrics,
        valid_metrics   = valid_metrics,
        tests_metrics   = tests_metrics,
        schedule        = args.schedule,
        monitor         = args.monitor,
        strict          = False,
    )

    trainer = Trainer(
        max_epochs              = args.max_epochs,
        accelerator             = args.accelerator,
        devices                 = args.devices,
        precision               = args.precision,
        log_every_n_steps       = args.log_every_n_steps,
        num_sanity_val_steps    = args.num_sanity_val_steps,
        deterministic           = args.deterministic,
        default_root_dir        = args.default_root_dir,
        callbacks               = callbacks,
        logger                  = logger,
        gradient_clip_val       = 0.5,
        gradient_clip_algorithm = 'value',
    )

    print("Train: %d | Valid: %d | Tests: %d" % (
        len(train_loader.dataset),
        len(valid_loader.dataset),
        len(tests_loader.dataset),
    ), file=sys.stderr)

    if args.train:
        if args.resume and checkpt:
            trainer.fit(trainee, train_loader, valid_loader, ckpt_path=checkpt)
        else:
            trainer.fit(trainee, train_loader, valid_loader)
    if args.validate:
        trainer.validate(trainee, dataloaders=valid_loader, ckpt_path=checkpt, verbose=False)
    if args.test:
        trainer.test(trainee, dataloaders=tests_loader, ckpt_path=checkpt, verbose=False)