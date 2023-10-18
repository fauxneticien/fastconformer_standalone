import hydra
# import jiwer
import torch
# import torchaudio

import lightning.pytorch as pl
# import pandas as pd

# from lhotse import CutSet, Fbank, FbankConfig
# from lhotse.dataset import AudioSamples
# from lhotse.dataset.collation import TokenCollater
# from lhotse.dataset.sampling import BucketingSampler
# from lhotse.recipes import download_librispeech, prepare_librispeech

from lightning import seed_everything
from lightning.pytorch.callbacks import LearningRateMonitor, ModelCheckpoint
# from lightning.pytorch.loggers import CSVLogger

# from torch.utils.data import DataLoader

from FastConformer.model import FastConformer, _init_conformer_params
from LibriSpeechFC import LibriSpeechDataModule

from lightning.pytorch.callbacks import ProgressBar

from tqdm import tqdm

torch.set_float32_matmul_precision("high")

class LhotseCompatibleProgressBar(ProgressBar):
    def __init__(self):
        super().__init__()  # don't forget this :)
        self.enable = True
        
        self.sanity_val_check_done = False
        self.sanity_val_check_steps = 0

    def disable(self):
        self.enable = False

    def on_sanity_check_end(self, trainer, pl_module):
        self.sanity_val_check_done = True

    def on_train_start(self, trainer, pl_module):
        self.train_pbar = tqdm(total=trainer.max_steps, dynamic_ncols=True)

    def on_train_epoch_start(self, trainer, pl_module):
        self.train_pbar.set_description_str(f"Epoch: {trainer.current_epoch}")

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        super().on_train_batch_end(trainer, pl_module, outputs, batch, batch_idx)  # don't forget this :)
        if self.sanity_val_check_done:
            self.train_pbar.set_postfix_str(",".join([ f"{k}: {v:.2f}" for (k,v) in outputs.items() ]))

    def on_before_optimizer_step(self, trainer, pl_module, optimizer):
        super().on_before_optimizer_step(trainer, pl_module, optimizer)  # don't forget this :)
        self.train_pbar.update(1)

    def on_train_end(self, trainer, pl_module):
        self.train_pbar.close()

    def on_validation_start(self, trainer, pl_module):
        if not self.sanity_val_check_done:
            self.val_pbar = tqdm(desc="Running full epoch to estimate number of validation batches...")
        else:
            self.val_pbar = tqdm(desc=f"Running validation", total=self.sanity_val_check_steps, dynamic_ncols=True)

    def on_validation_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        super().on_validation_batch_end(trainer, pl_module, outputs, batch, batch_idx)  # don't forget this :)

        if not self.sanity_val_check_done:
            self.sanity_val_check_steps += 1
        else:
            self.val_pbar.update(1)

    def on_validation_end(self, trainer, pl_module):
        self.val_pbar.close()

# train_lnl.py imports this from models/DeepSpeech/lightningmodule.py
class FastConformerLightningModule(pl.LightningModule):

    def __init__(self, cfg, n_feature=80):
        super().__init__()
        self.cfg = cfg

        # Set input feature dimension on init so DataModule can use info
        self.n_feature = n_feature

        self.val_losses = []
        self.val_ref_pred_pairs = []

    def setup(self, stage = None):

        self.cfg.encoder.num_labels = 500

        self.model = FastConformer(**self.cfg.encoder)
        self.model.apply(_init_conformer_params)
        
        self.ce_loss = torch.nn.CrossEntropyLoss(ignore_index=-1, reduction="sum")

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.cfg.lr)
        scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=self.cfg.lr, total_steps=self.cfg.max_updates, anneal_strategy='linear')

        return {
            "optimizer": optimizer,
            "lr_scheduler": {"scheduler": scheduler, "interval": "step", 'frequency': 1 }
        }

    def pretrain_forward(self, feats_padded, feats_lens):
        preenc_feats, preenc_lens = self.model.pre_encode(feats_padded, feats_lens)

        max_audio_length=preenc_feats.size(1)
        self.model.pos_enc.extend_pe(max_audio_length, preenc_feats.device)

        preenc_feats, pos_emb = self.model.pos_enc(preenc_feats)

        pad_mask, att_mask = self.model._create_masks(
            att_context_size=[-1, -1] if self.model.self_attention_model == 'rel_pos' else [128,128],
            padding_length=preenc_lens,
            max_audio_length=max_audio_length,
            offset=None,
            device=preenc_feats.device,
            self_attention_model=self.model.self_attention_model
        )

        for i, layer in enumerate(self.model.layers):
            enc_feats = layer(
                x=preenc_feats if i == 0 else enc_feats,
                att_mask=att_mask,
                pos_emb=pos_emb,
                pad_mask=pad_mask,
                cache_last_channel=None,
                cache_last_time=None,
            )

        decoder_output = self.model.decoder.decoder_layers(
            enc_feats.transpose(1,2)
        ).transpose(1, 2)

        return decoder_output

    def _step(self, batch, batch_idx, step_type):
        # For details, see loss usage section in notebooks/01_mwe.ipynb
        outputs = self.pretrain_forward(batch['feats_padded'], batch['feats_lens'])
        loss = self.ce_loss(outputs.transpose(1,2), batch["ptlabels_padded"])

        # For multi-GPU training, normalize the loss based on the sum of batch_size across all GPUs
        batch_size = batch['feats_padded'].size(0)
        # Get batch sizes from all GPUs
        batch_sizes = self.all_gather(batch_size)
        # Normalize by world size / batch size
        loss *= batch_sizes.size(0) / batch_sizes.sum()

        self.log(f"{step_type}/loss", loss.item(), sync_dist=True, prog_bar=True)

        return loss

    def training_step(self, batch, batch_idx):
        loss = self._step(batch, batch_idx, "train")
        return loss
    
    def validation_step(self, batch, batch_idx):
        loss = self._step(batch, batch_idx, "val")
        return loss

@hydra.main(version_base="1.3", config_path="./config", config_name="default_arch")
def train(cfg):

    seed_everything(42)

    # train_lnl.py imports an over-ridable configuration merged
    # from various .yaml files in configs/ using Hydra (https://hydra.cc/docs/intro/)
    LNL_CONFIG = {
        "lr": 1e-4,
        "max_updates": 10_000,
        "grad_acc": 1,
        "val_every_n_updates": 5_000,
    }

    checkpoint_callback = ModelCheckpoint(
        dirpath='checkpoints/',
        filename='fastconformer-v0-pretrain_{step}',
        # Keep all checkpoints saved at every_n_train_steps
        save_top_k=-1,
        # n_train_steps in this case does correspond to number of optimizer steps
        every_n_train_steps=cfg.checkpoint_every_n_updates,
        verbose=True
    )

    wandb_logger = pl.loggers.WandbLogger(
        project="fastconformer_finetuning",
        name="fastconformer_selfsupervised_dev_400k"
    )

    trainer = pl.Trainer(
        precision="16-mixed",
        max_steps=cfg.max_updates,
        accumulate_grad_batches=LNL_CONFIG["grad_acc"],
        max_epochs=-1,
        # Turns out val_check_interval is based on dataloader batch steps not update steps
        # Disable validation after every epoch then set validation to occurr after every N updates
        check_val_every_n_epoch=None,
        val_check_interval=LNL_CONFIG["val_every_n_updates"] * LNL_CONFIG["grad_acc"],
        # Use DDP and prevent Lightning from replacing Lhotse's DDP-compatible sampler
        strategy="ddp",
        use_distributed_sampler=False,
        callbacks=[ 
            LearningRateMonitor(logging_interval='step'),
            LhotseCompatibleProgressBar(),
            checkpoint_callback
        ],
        # Disabled for this minimal working example
        enable_model_summary=False,
        logger=wandb_logger
    )

    trainer.fit(
        FastConformerLightningModule(cfg),
        LibriSpeechDataModule(
            "pretrain",
            "train",
            "dev",
            40,
            train_max_dur=1600,
            dev_max_dur=1600,
            corpus_dir="./data/LibriSpeech_all",
            num_dl_workers=4
        )
    )

if __name__ == "__main__":
    train()
