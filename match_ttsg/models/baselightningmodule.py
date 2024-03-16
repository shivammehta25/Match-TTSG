"""
This is a base lightning module that can be used to train a model.
The benefit of this abstraction is that all the logic outside of model definition can be reused for different models.
"""
import inspect
from abc import ABC
from typing import Any, Dict

import torch
from lightning import LightningModule
from lightning.pytorch.utilities import grad_norm

from match_ttsg import utils
from match_ttsg.utils.model import denormalize, normalize
from match_ttsg.utils.utils import plot_tensor

log = utils.get_pylogger(__name__)


class BaseLightningClass(LightningModule, ABC):
    def update_data_statistics(self, data_statistics):
        if data_statistics is None:
           raise ValueError(f"data_statistics are not computed. \
                             Please run python match_ttsg/utils/preprocess_data.py -i <dataset.yaml> \
                             to get statistics and update them in data_statistics field.") 

        self.register_buffer('mel_mean', torch.tensor(data_statistics['mel_mean']))
        self.register_buffer('mel_std', torch.tensor(data_statistics['mel_std']))
        
        self.register_buffer('motion_mean', torch.tensor(data_statistics['motion_mean']))
        self.register_buffer('motion_std', torch.tensor(data_statistics['motion_std']))
        
        self.register_buffer("pitch_mean", torch.tensor(data_statistics["pitch_mean"]))
        self.register_buffer("pitch_std", torch.tensor(data_statistics["pitch_std"]))
        
        self.register_buffer("energy_mean", torch.tensor(data_statistics["energy_mean"]))
        self.register_buffer("energy_std", torch.tensor(data_statistics["energy_std"]))
        
        
        pitch_min = normalize(torch.tensor(data_statistics["pitch_min"]), self.pitch_mean, self.pitch_std)
        pitch_max = normalize(torch.tensor(data_statistics["pitch_max"]), self.pitch_mean, self.pitch_std)
        energy_min = normalize(torch.tensor(data_statistics["energy_min"]), self.energy_mean, self.energy_std)
        energy_max = normalize(torch.tensor(data_statistics["energy_max"]), self.energy_mean, self.energy_std)
        
        self.register_buffer("pitch_min", pitch_min)
        self.register_buffer("pitch_max", pitch_max)
        self.register_buffer("energy_min", energy_min)
        self.register_buffer("energy_max", energy_max)

    def configure_optimizers(self) -> Any:
        optimizer = self.hparams.optimizer(params=self.parameters())
        if self.hparams.scheduler not in (None, {}):
            scheduler_args = {}
            # Manage last epoch for exponential schedulers
            if "last_epoch" in inspect.signature(self.hparams.scheduler.scheduler).parameters:
                if hasattr(self, "ckpt_loaded_epoch"):
                    current_epoch = self.ckpt_loaded_epoch - 1
                else:
                    current_epoch = -1

            scheduler_args.update({"optimizer": optimizer})
            scheduler = self.hparams.scheduler.scheduler(**scheduler_args)
            scheduler.last_epoch = current_epoch
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "interval": self.hparams.scheduler.lightning_args.interval,
                    "frequency": self.hparams.scheduler.lightning_args.frequency,
                    "name": "learning_rate",
                },
            }

        return {"optimizer": optimizer}

    def get_losses(self, batch):
        x, x_lengths = batch["x"], batch["x_lengths"]
        y, y_lengths = batch["y"], batch["y_lengths"]
        spks = batch["spks"]
        y_motion = batch["y_motion"]
        

        dur_loss, prior_loss, diff_loss = self(
            x=x,
            x_lengths=x_lengths,
            y=y,
            y_lengths=y_lengths,
            y_motion=y_motion,
            spks=spks,
            out_size=self.out_size,
        )
        return {
            "dur_loss": dur_loss,
            "prior_loss": prior_loss,
            "diff_loss": diff_loss,
        }

    def on_load_checkpoint(self, checkpoint: Dict[str, Any]) -> None:
        self.ckpt_loaded_epoch = checkpoint["epoch"]  # pylint: disable=attribute-defined-outside-init

    def training_step(self, batch: Any, batch_idx: int):
        loss_dict = self.get_losses(batch)
        self.log(
            "step",
            float(self.global_step),
            on_step=True,
            prog_bar=True,
            logger=True,
            sync_dist=True,
        )

        self.log(
            "sub_loss/train_dur_loss",
            loss_dict["dur_loss"],
            on_step=True,
            on_epoch=True,
            logger=True,
            sync_dist=True,
        )
        self.log(
            "sub_loss/train_prior_loss",
            loss_dict["prior_loss"],
            on_step=True,
            on_epoch=True,
            logger=True,
            sync_dist=True,
        )
        self.log(
            "sub_loss/train_diff_loss",
            loss_dict["diff_loss"],
            on_step=True,
            on_epoch=True,
            logger=True,
            sync_dist=True,
        )

        total_loss = sum(loss_dict.values())
        self.log(
            "loss/train",
            total_loss,
            on_step=True,
            on_epoch=True,
            logger=True,
            prog_bar=True,
            sync_dist=True,
        )

        return {"loss": total_loss, "log": loss_dict}

    def validation_step(self, batch: Any, batch_idx: int):
        loss_dict = self.get_losses(batch)
        self.log(
            "sub_loss/val_dur_loss",
            loss_dict["dur_loss"],
            on_step=True,
            on_epoch=True,
            logger=True,
            sync_dist=True,
        )
        self.log(
            "sub_loss/val_prior_loss",
            loss_dict["prior_loss"],
            on_step=True,
            on_epoch=True,
            logger=True,
            sync_dist=True,
        )
        self.log(
            "sub_loss/val_diff_loss",
            loss_dict["diff_loss"],
            on_step=True,
            on_epoch=True,
            logger=True,
            sync_dist=True,
        )

        total_loss = sum(loss_dict.values())
        self.log(
            "loss/val",
            total_loss,
            on_step=True,
            on_epoch=True,
            logger=True,
            prog_bar=True,
            sync_dist=True,
        )

        return total_loss

    def on_validation_end(self) -> None:
        if self.trainer.is_global_zero:
            one_batch = next(iter(self.trainer.val_dataloaders))
            if self.current_epoch == 0:
                log.debug("Plotting original samples")
                for i in range(2):
                    y = one_batch["y"][i].unsqueeze(0).to(self.device)
                    self.logger.experiment.add_image(
                        f"original/{i}",
                        plot_tensor(y.squeeze().cpu()),
                        self.current_epoch,
                        dataformats="HWC",
                    )

            log.debug("Synthesising...")
            for i in range(2):
                x = one_batch["x"][i].unsqueeze(0).to(self.device)
                x_lengths = one_batch["x_lengths"][i].unsqueeze(0).to(self.device)
                spks = one_batch["spks"][i].unsqueeze(0).to(self.device) if one_batch["spks"] is not None else None
                output = self.synthesise(x[:, :x_lengths], x_lengths, n_timesteps=10, spks=spks)
                y_enc, y_dec = output["encoder_outputs_mel"], output["decoder_outputs_mel"]
                y_motion_enc, y_motion_dec, attn = output['encoder_outputs_motion'], output['decoder_outputs_motion'], output['attn']
                attn = output["attn"]
                self.logger.experiment.add_image(
                    f"generated_enc/mel_{i}",
                    plot_tensor(y_enc.squeeze().cpu()),
                    self.current_epoch,
                    dataformats="HWC",
                )
                self.logger.experiment.add_image(
                    f"generated_dec/mel_{i}",
                    plot_tensor(y_dec.squeeze().cpu()),
                    self.current_epoch,
                    dataformats="HWC",
                )
                self.logger.experiment.add_image(
                    f"alignment/{i}",
                    plot_tensor(attn.squeeze().cpu()),
                    self.current_epoch,
                    dataformats="HWC",
                )
                self.logger.experiment.add_image(f'generated_enc/motion_{i}', plot_tensor(y_motion_enc.squeeze().cpu()), self.current_epoch, dataformats='HWC')
                self.logger.experiment.add_image(f'generated_dec/motion_{i}', plot_tensor(y_motion_dec.squeeze().cpu()), self.current_epoch, dataformats='HWC')

    def on_before_optimizer_step(self, optimizer):
        self.log_dict({f"grad_norm/{k}": v for k, v in grad_norm(self, norm_type=2).items()})
