import os, sys
from opt import get_opts
import torch
from collections import defaultdict
import wandb
from torch.utils.data import DataLoader
from datasets import dataset_dict
from eval import eval

from rich import print
from rich import pretty
pretty.install()
from rich import traceback
traceback.install()

# models
from models.nerf import Embedding, NeRF
from models.rendering import render_rays

# optimizer, scheduler, visualization
from utils import *

# losses
from losses import loss_dict

# metrics
from metrics import *

# pytorch-lightning
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning import LightningModule, Trainer

class NeRFSystem(LightningModule):
    def __init__(self, hparams):
        super(NeRFSystem, self).__init__()
        self.hparams = hparams

        self.loss = loss_dict[hparams.loss_type]()

        self.embedding_xyz = Embedding(3, 10) # 10 is the default number
        self.embedding_dir = Embedding(3, 4) # 4 is the default number
        self.embeddings = [self.embedding_xyz, self.embedding_dir]

        self.nerf_coarse = NeRF()
        self.models = [self.nerf_coarse]
        if hparams.N_importance > 0:
            self.nerf_fine = NeRF()
            self.models += [self.nerf_fine]

        for m in self.models:
            wandb.watch(m, log="all", log_freq=100)

    def decode_batch(self, batch):
        rays = batch['rays'] # (B, 8)
        rgbs = batch['rgbs'] # (B, 3)
        return rays, rgbs

    def forward(self, rays):
        """Do batched inference on rays using chunk."""
        B = rays.shape[0]
        results = defaultdict(list)
        for i in range(0, B, self.hparams.chunk):
            rendered_ray_chunks = \
                render_rays(self.models,
                            self.embeddings,
                            rays[i:i+self.hparams.chunk],
                            self.hparams.N_samples,
                            self.hparams.use_disp,
                            self.hparams.perturb,
                            self.hparams.noise_std,
                            self.hparams.N_importance,
                            self.hparams.chunk, # chunk size is effective in val mode
                            self.train_dataset.white_back)

            for k, v in rendered_ray_chunks.items():
                results[k] += [v]

        for k, v in results.items():
            results[k] = torch.cat(v, 0)
        return results

    def prepare_data(self):
        dataset = dataset_dict[self.hparams.dataset_name]
        kwargs = {'root_dir': self.hparams.root_dir,
                  'img_wh': tuple(self.hparams.img_wh)}
        if self.hparams.dataset_name == 'llff':
            kwargs['spheric_poses'] = self.hparams.spheric_poses
            kwargs['val_num'] = self.hparams.num_gpus
        self.train_dataset = dataset(split='train', **kwargs)
        self.val_dataset = dataset(split='val', **kwargs)

    def configure_optimizers(self):
        self.optimizer = get_optimizer(self.hparams, self.models)
        scheduler = get_scheduler(self.hparams, self.optimizer)
        
        return [self.optimizer], [scheduler]

    def train_dataloader(self):
        return DataLoader(self.train_dataset,
                          shuffle=True,
                          num_workers=4,
                          batch_size=self.hparams.batch_size,
                          pin_memory=True)

    def val_dataloader(self):
        return DataLoader(self.val_dataset,
                          shuffle=False,
                          num_workers=4,
                          batch_size=1, # validate one image (H*W rays) at a time
                          pin_memory=True)
    
    def training_step(self, batch, batch_nb):
        log = {'lr': get_learning_rate(self.optimizer)}
        rays, rgbs = self.decode_batch(batch)
        results = self(rays)
        log['train/loss'] = loss = self.loss(results, rgbs)
        typ = 'fine' if 'rgb_fine' in results else 'coarse'

        with torch.no_grad():
            psnr_ = psnr(results[f'rgb_{typ}'], rgbs)
            log['train/psnr'] = psnr_

        if self.global_step % 10 == 0:
            wandb.log(log, step=self.global_step)

        return {'loss': loss,
                'progress_bar': {'train_psnr': psnr_},
                # 'log': log
               }

    def validation_step(self, batch, batch_nb):
        rays, rgbs = self.decode_batch(batch)
        rays = rays.squeeze() # (H*W, 3)
        rgbs = rgbs.squeeze() # (H*W, 3)
        results = self(rays)
        log = {'val_loss': self.loss(results, rgbs)}
        typ = 'fine' if 'rgb_fine' in results else 'coarse'
    
        if batch_nb == 0:
            W, H = self.hparams.img_wh
            img = results[f'rgb_{typ}'].view(H, W, 3).cpu()
            img = img.permute(2, 0, 1) # (3, H, W)
            img_gt = rgbs.view(H, W, 3).permute(2, 0, 1).cpu() # (3, H, W)
            depth = visualize_depth(results[f'depth_{typ}'].view(H, W)) # (3, H, W)
            stack = torch.stack([img_gt, img, depth]) # (3, 3, H, W)
            wandb.log({'val/GT_pred_depth':wandb.Image(stack)})


        log['val_psnr'] = psnr(results[f'rgb_{typ}'], rgbs)
        wandb.log(log)
        return log

    def validation_epoch_end(self, outputs):
        mean_loss = torch.stack([x['val_loss'] for x in outputs]).mean()
        mean_psnr = torch.stack([x['val_psnr'] for x in outputs]).mean()

        log = {'val/loss': mean_loss,
         'val/psnr': mean_psnr}
        wandb.log(log)
        return {'progress_bar': {'val_loss': mean_loss,
                                 'val_psnr': mean_psnr},
                # 'log': log
               }

    def on_fit_end(self):
        print("completed training and running evaluation")
        ckpt_dir = os.path.join(self.hparams.log_dir, self.hparams.exp_name, "ckpts")
        wandb.save(ckpt_dir +"/*")

        self.hparams.scene_name = self.hparams.exp_name
        self.hparams.N_importance=64

        ckpts = [f for f in os.listdir(ckpt_dir) if "epoch" in f]
        ckpts.sort()
        # if len(ckpts) != 0:
        self.hparams.eval_ckpt_path =os.path.join(ckpt_dir, ckpts[-1])
        eval_out_path = eval(self.hparams)

        for file_name in os.listdir(eval_out_path):
            if file_name.endswith("gif"):
                wandb.log({"eval/gif":wandb.Video(os.path.join(eval_out_path, file_name), fps=30, format="gif")})

if __name__ == '__main__':
    hparams = get_opts()

    tags = ["train"]
    if hparams.debug:
        tags.append("debug")
    wandb.init(name=hparams.exp_name, dir="/root/nerf_pl/", project="nerf", entity="stereo",
               save_code=True, job_type="train", tags=tags)

    wandb.config.update(hparams)

    if hparams.save_dataset and not hparams.debug:
        print("saving dataset artifact...")
        dataset_name = hparams.root_dir.split("/")[-1]
        artifact = wandb.Artifact(dataset_name, type="dataset")
        artifact.add_dir(hparams.root_dir)
        artifact.description = hparams.exp_name
        wandb.log_artifact(artifact)

    system = NeRFSystem(hparams)
    checkpoint_callback = ModelCheckpoint(filepath=os.path.join(hparams.log_dir, hparams.exp_name, 'ckpts',
                                                                '{epoch:d}'),
                                          monitor='val/loss',
                                          mode='min',
                                          save_top_k=5,save_last=True,)



    # logger = TestTubeLogger(
    #     save_dir=hparams.log_dir,
    #     name=hparams.exp_name,
    #     debug=False,
    #     create_git_tag=False
    # )
    #
    # logger = WandbLogger(name=hparams.exp_name,
    #                           save_di=os.path.join(hparams.log_dir, hparams.exp_name),
    #                           project="nerf")

    trainer = Trainer(max_epochs=hparams.num_epochs,
                      checkpoint_callback=checkpoint_callback,
                      resume_from_checkpoint=hparams.ckpt_path,
                      weights_summary=None,
                      progress_bar_refresh_rate=1,
                      gpus=hparams.num_gpus,
                      distributed_backend='ddp' if hparams.num_gpus>1 else None,
                      num_sanity_val_steps=1,
                      benchmark=True,
                      profiler=hparams.num_gpus==1)

    trainer.fit(system)

    if not hparams.debug:
        print("saving dataset artifact...")
        ckpt_dir = os.path.join(hparams.log_dir, hparams.exp_name, "ckpts")
        artifact = wandb.Artifact(hparams.exp_name, type="model")
        artifact.add_dir(ckpt_dir)
        artifact.description = hparams.exp_name
        wandb.log_artifact(artifact)

    wandb.finish()