# python
import os
import re
import time
import logging
import numpy as np
import pickle as pkl
import pandas as pd
from typing import List, Union, Tuple
# torch
import torch
import torchvision
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from prime_evaluator.data_type import get_datatype_by_name
from models import get_model_by_name
from losses import get_loss_by_name

from utils.config import (
    _MAX_GUESSES_NUM, _MISSING_THRESHOLD, _HISTOGRAM_BINS
)
from prime_evaluator.utils.calcu_utils import AverageMeter
import prime_evaluator.utils.learning_utils as learning_utils
from prime_evaluator.utils.visual import get_viz_by_name
from prime_evaluator.utils.eval_forecasting import (
    get_accuracy_statistic,
    get_fde_statistic,
    compute_forecasting_metrics
)
# from argoverse.evaluation.competition_util import generate_forecasting_h5


class Predictor():
    def __init__(self, args):
        self.args = args
        self.save_dir = os.path.abspath(
            os.path.join(args.ckptdir, args.subdir or args.mode)
        )
        os.makedirs(self.save_dir, exist_ok=True)
        ## Logging
        logging.basicConfig(filename=f'{self.save_dir}/logging.log',
                            filemode='a',
                            level=getattr(logging, args.logging.upper(), None),
                            format='[%(levelname)s %(asctime)s] %(message)s',
                            datefmt='%m-%d %H:%M:%S')
        logging.getLogger().addHandler(logging.StreamHandler())

        self.use_cuda = args.use_cuda
        self.net_name = args.net_name
        self.create_dataloader(args)
        self.create_model(args)
        self.create_loss(args)
        self.create_optimizer(args)
        self.create_visualizer(args)
        self.epoch = 0
        self.train_step = 0
        self.val_step = 0
        self.best_loss = np.inf
        # save results
        self.pred_trajs = {}
        self.pred_prbs = {}
        self.gt_trajs = {}
        self.cities = {}
        ## Init some parameters listed above
        self.init_predictor(args)

        if args.tensorboard:
            if args.tensorboard_dir:
                self.writer = SummaryWriter(args.tensorboard_dir)
            else:
                self.writer = SummaryWriter(self.save_dir)
        logging.info(f"\n=================================== RUNNING!!! ===================================")
        logging.info(
            f"[{args.mode} FROM] epoch: {self.epoch}, step: {self.train_step}, best loss: {self.best_loss}")


    def create_dataloader(self, args):
        logging.info(f"\n=================================== Datasets ({args.mode} mode) =========================")
        logging.info(f"FlagDataset: {args.flag_dataset}")
        if args.mode in ['test', 'val', 'submit']:
            logging.info(f"Test dataset {args.dataset}.")
            # eval model
            self.dataset_val = get_datatype_by_name(args.flag_dataset, args.dataset_val, args, augment=False)
            self.dataloader_val = DataLoader(self.dataset_val,
                                             batch_size=args.batch_size,
                                             shuffle=False,
                                             num_workers=args.num_workers,
                                             collate_fn=self.dataset_val.collate_fn,
                                             pin_memory=True)
        else:
            logging.info(f"Train dataset: {args.dataset}.")
            self.dataset_train = get_datatype_by_name(args.flag_dataset, args.dataset, args, args.data_augment)
            self.dataloader_train = DataLoader(self.dataset_train,
                                               batch_size=args.batch_size,
                                               shuffle=True,
                                               num_workers=args.num_workers,
                                               collate_fn=self.dataset_train.collate_fn,
                                               pin_memory=False, drop_last=True)

            logging.info(f"Val dataset: {args.dataset_val}.")
            self.dataset_val = get_datatype_by_name(args.flag_dataset, args.dataset_val, args, augment=False)
            self.dataloader_val = DataLoader(self.dataset_val,
                                             batch_size=args.batch_size,
                                             shuffle=False,
                                             num_workers=args.num_workers,
                                             collate_fn=self.dataset_val.collate_fn,
                                             pin_memory=False)

    def create_model(self, args):
        logging.info(f"\n=================================== Network ({args.net_name}) =========================")
        self.model = get_model_by_name(args.net_name, args, device=torch.device("cuda:{}".format(args.rank) if torch.cuda.is_available() else "cpu"))
        logging.info(f"\n{self.model}\nTotal parameters: {sum(param.numel() for param in self.model.parameters())}")
        if args.use_cuda:
            self.model = self.model.cuda()


    def create_loss(self, args):
        logging.info(f"\n=================================== Loss ({args.loss_name}) =========================")
        self.lossfun = get_loss_by_name(args.loss_name, args)


    def create_optimizer(self, args):
        logging.info(f"\n=================================== Learning Params =========================")
        logging.info(
            f'Optimizer: {args.optim}, LR: {args.learning_rate}, Batch: {args.batch_size}, Epochs: {args.num_epochs}')
        if args.lr_decay_epochs:
            logging.info(f'Learning Rate decays by {args.lr_decay_factor} at epochs {args.lr_decay_epochs}')
        if args.optim.lower() == 'adam':
            self.optim = torch.optim.Adam(self.model.parameters(), lr=args.learning_rate)
        elif args.optim.lower() == 'sgd':
            self.optim = torch.optim.SGD(self.model.parameters(), lr=args.learning_rate)
        else:
            assert False, "not optimizer specified."


    def create_visualizer(self, args):
        self.visualizer = get_viz_by_name(args.viz_name, args)


    def init_predictor(self, args):
        """initialize predictor by given pretained weight (--path_weight) or checkpoint (self.save_dir)
        """
        if args.mode == 'train':
            self.load_prev_ckpt()

        elif args.mode == 'finetune':
            self.load_prev_ckpt(best=True)

        elif args.mode in ['val', 'submit', 'test']:
            if os.path.exists(args.path_weight):
                # ONLY load the pretrained weight
                logging.info(f"Pretrained WEIGHTS loaded from {args.path_weight}")
                state_dict = torch.load(args.path_weight)['state_dict']
                self.model.load_state_dict(state_dict)
            else:
                assert False, f"No pretrained weights could be loaded from {args.path_weight}"

        else:
            logging.info("##### OTHER MODES #####")


    def load_prev_ckpt(self, best=False):
        # Loading ckpt from the best or the lastest status (model).
        logging.info(f"Try to load ckpt from dir {self.save_dir}. {'BEST' if best else 'LATEST'}.")
        state = learning_utils.load_checkpoint(self.save_dir, best)
        if state is None:
            logging.info("No checkpoint is loaded and train from scratch.")
        else:
            self.epoch = state['epoch'] + 1
            self.train_step = state['train_step'] + 1
            self.best_loss = state['best_loss']
            self.model.load_state_dict(state['state_dict'])
            self.optim.load_state_dict(state['optim'])


    def save_cur_ckpt(self, epoch, best_loss, is_best):
        state = {
            'epoch': epoch,
            'train_step': self.train_step,
            'best_loss': best_loss,
            'state_dict': self.model.state_dict(),
            'optim': self.optim.state_dict(),
            'args': self.args,
        }
        # this checkpoint will be overwritten every save, i.e., the latest
        learning_utils.save_checkpoint(state, is_best, dirpath=self.save_dir, filename='model_checkpoint.pth')
        # this checkpoint will be backup for this epoch
        learning_utils.save_checkpoint(state, is_best, dirpath=self.save_dir, filename=f'model_checkpoint_{str(epoch)}.pth')
        if is_best:
            path_save = os.path.join(self.save_dir, 'weight_best.pth')
            if torch.__version__ >= '1.6.0':
                torch.save({'state_dict': self.model.state_dict()}, path_save, _use_new_zipfile_serialization=False)
            else:
                torch.save({'state_dict': self.model.state_dict()}, path_save)

    def start(self):
        args = self.args
        if args.mode in ['train', 'finetune']:
            ## Epoches Loop
            init_epoch = self.epoch
            logging.info(f"starting from {init_epoch} << total {args.num_epochs}")

            for epoch in range(init_epoch, args.num_epochs):
                self.epoch = epoch
                train_loss = self.train()
                ## Save the best (in validation) & last model for recovering.
                if (epoch % args.val_every_n_epoch == 0) or (epoch == args.num_epochs - 1):
                    # evaluate on validation set
                    loss_epoch = self.validate()
                    is_best = loss_epoch < self.best_loss
                    self.best_loss = min(loss_epoch, self.best_loss)
                    self.save_cur_ckpt(epoch, self.best_loss, is_best)

        elif args.mode == 'val':
            self.validate()
        elif args.mode == 'submit':
            self.submit()
        elif args.mode == 'test':
            self.test()
        else:
            assert False, f"Wrong type of mode {args.mode} is specified [train/finetune/val/submit/test]"


    def train(self):
        args = self.args
        losses = AverageMeter()
        data_time = AverageMeter()
        batch_time = AverageMeter()
        post_time = AverageMeter()

        logging.info(f"\n******************** [TRAIN] Epoch {self.epoch}/{args.num_epochs} ********************")
        learning_utils.adjust_learning_rate(self.optim, self.epoch, args.lr_decay_epochs, args.learning_rate, args.lr_decay_factor)
        self.model.train()

        time_end = time.time()
        for i_batch, (sample_valid, sample_batched) in enumerate(self.dataloader_train):
            ## Preprocess
            data_time.update(time.time()-time_end)
            if not sample_valid:
                logging.warning("Invalid batched sample encountered in training")
                continue
            if args.use_cuda:
                for key in sample_batched:
                    if torch.is_tensor(sample_batched[key]):
                        sample_batched[key] = sample_batched[key].cuda()
            ## Network Propagation & update
            self.optim.zero_grad()
            output = self.model(sample_batched, mode="val")
            loss = self.lossfun(output, sample_batched)
            loss.backward()
            self.optim.step()
            self.train_step += 1

            losses.update(loss.item())
            batch_time.update(time.time()-time_end)

            ## Postprocess
            processed_output, sample_batched = \
                self.model.post_process(self.dataset_train, output, sample_batched, phase='train')
            post_time.update(time.time()-time_end)
            time_end = time.time()

            if i_batch % self.args.print_every_n_steps == 0:
                logging.info(
                    f"[train] epoch {self.epoch}/{args.num_epochs}\tbatch {i_batch}/{len(self.dataloader_train)}\t"
                    f"loss {losses.val:.3f}\tavg {losses.avg:.3f}"
                )
                self.writer.add_scalar('LossTrain/step', loss.item(), self.train_step)

        ## Complete 1 epoch
        if args.tensorboard:
            self.writer.add_scalar('LossTrain/epoch', losses.avg, self.epoch)

        return losses.avg


    def validate(self):
        args = self.args
        losses = AverageMeter()
        metric_recorder = None
        fde_statistic = []
        logging.info(f"******************** [VALIDATION] Epoch {self.epoch}/{args.num_epochs} ********************")

        self.model.eval()
        with torch.no_grad():
            for i_batch, (sample_valid, sample_batched) in enumerate(self.dataloader_val):
                if not sample_valid:
                    logging.warning("Invalid batched sample encountered in validation")
                    continue
                if args.use_cuda:
                    for key in sample_batched:
                        if torch.is_tensor(sample_batched[key]):
                            sample_batched[key] = sample_batched[key].cuda()

                output = self.model(sample_batched, mode="val")
                loss = self.lossfun(output, sample_batched)
                losses.update(loss.item())
                self.val_step += 1

                # Accuracy metric calculation
                filter_output, filter_prbs, nofilter_output, sample_batched = \
                    self.model.post_process(self.dataset_val, output, sample_batched, phase='validate')
                metrics = get_accuracy_statistic(self.dataset_val, self.args.pred_len, nofilter_output, sample_batched)
                metrics_flt = get_accuracy_statistic(self.dataset_val, self.args.pred_len, filter_output, sample_batched,
                                                     fixed_guesses_num=_MAX_GUESSES_NUM,
                                                     postfix=f'_THRE{str(self.model.preds_filter.endpt_dist)}')
                fde_list = get_fde_statistic(self.dataset_val, self.args.pred_len, filter_output, sample_batched)
                fde_statistic = fde_statistic + fde_list

                # Create the metric recorder and update it
                if metric_recorder is None:
                    metric_recorder = {}
                    for k in metrics:
                        metric_recorder[k] = AverageMeter()
                    for k in metrics_flt:
                        metric_recorder[k] = AverageMeter()
                for k in metrics:
                    metric_recorder[k].update(metrics[k], len(nofilter_output))
                for k in metrics_flt:
                    metric_recorder[k].update(metrics_flt[k], len(nofilter_output))

                # Loss recording
                if (i_batch % self.args.print_every_n_steps == 0) or (i_batch + 1 == len(self.dataloader_val)):
                    logging.info(
                        f"[val] epoch {self.epoch}/{args.num_epochs}\tbatch {i_batch}/{len(self.dataloader_val)}\t"
                        f"loss {losses.val:.3f}\tavg {losses.avg:.3f}"
                    )
                    self.writer.add_scalar('LossVal/step', loss.item(), self.val_step)

                # Images recording in tensorboard (second last batch)
                if i_batch+2==len(self.dataloader_val) and args.tensorboard and self.visualizer:
                    debug_imgs = self.visualizer.render(output, sample_batched, filter_output, max_n=16)
                    img_grid = torchvision.utils.make_grid(torch.from_numpy(debug_imgs), nrow=1)
                    self.writer.add_image('ValidSample/epoch_imgs', img_grid, self.epoch)

                if args.plot:
                    assert args.plot in ['fde', 'all'], "Unsupported plot option."
                    self.visualizer.render_and_save(output, sample_batched, filter_output, max_n=16)    # Plot output with filter

                if args.save:
                    self.pred_trajs.update(filter_output)
                    self.pred_prbs.update(filter_prbs)
                    gts, cities = self.dataset_val.get_ground_truth(sample_batched)
                    self.gt_trajs.update(gts)
                    self.cities.update(cities)

            # Complete the whole evaluation process.
            for i, k in enumerate(metric_recorder):
                post_fix = "" if i % 5 else " --------------------"
                logging.info(f"[metric] {k} = {metric_recorder[k].avg:.3f}{post_fix}")
            fde_counts, fde_bins = np.histogram(fde_statistic, bins=_HISTOGRAM_BINS)
            logging.info(f"[stat] Bins: {fde_bins}\n[stat] Count: {fde_counts}")

            if args.tensorboard:
                self.writer.add_scalar('LossVal/epoch', losses.avg, self.epoch)
                self.writer.add_histogram(f'FdeStat_THRE{str(self.model.preds_filter.endpt_dist)}/epoch', np.array(fde_statistic), self.epoch)
                for k in metric_recorder:
                    k_name = learning_utils.to_leagal_tf_summary_name(k)
                    self.writer.add_scalar(k_name, metric_recorder[k].avg, self.epoch)
                self.writer.flush()

            if args.save:
                metrics_6 = compute_forecasting_metrics(self.pred_trajs, self.gt_trajs, self.cities, 6, args.pred_len, self.pred_prbs)
                metrics_1 = compute_forecasting_metrics(self.pred_trajs, self.gt_trajs, self.cities, 1, args.pred_len, self.pred_prbs)
                result = dict(preds=self.pred_trajs,
                              probs=self.pred_prbs,
                              gts=self.gt_trajs,
                              cities=self.cities,
                              metrics_6=metrics_6,
                              metrics_1=metrics_1,
                              )
                with open(os.path.abspath(os.path.join(self.args.ckptdir, self.args.subdir or self.args.mode, "result.pkl")), "wb") as f:
                    pkl.dump(result, f)

        return losses.avg


    def submit(self):
        args = self.args

        self.model.eval()
        with torch.no_grad():
            for i_batch, (sample_valid, sample_batched) in enumerate(self.dataloader_val):
                if not sample_valid:
                    continue

                if args.use_cuda:
                    for key in sample_batched:
                        if torch.is_tensor(sample_batched[key]):
                            sample_batched[key] = sample_batched[key].cuda()

                output = self.model(sample_batched)
                processed_output, processed_prb = \
                    self.model.post_process(self.dataset_val, output, sample_batched, phase='submit')

                if isinstance(processed_output, dict):
                    self.pred_trajs.update(processed_output.copy())
                    self.pred_prbs.update(processed_prb.copy())

                if i_batch % self.args.print_every_n_steps == 0:
                    logging.info(
                        f"[submit] epoch {self.epoch}/{args.num_epochs} batch {i_batch}/{len(self.dataloader_val)} num_seqs {len(processed_output)}"
                    )

        with open(f'{self.save_dir}/pred_trajs_submit.pkl', "wb") as f:
            pkl.dump(self.pred_trajs, f)
            logging.info(f"[submit] Save predicted trajecoty in {self.save_dir}/pred_trajs_submit.pkl")
        with open(f'{self.save_dir}/pred_prbs_submit.pkl', "wb") as f:
            pkl.dump(self.pred_prbs, f)
            logging.info(f"[submit] Save predicted probability in {self.save_dir}/pred_prbs_submit.pkl")

        h5file_name = time.strftime("%m%d-%H%M-%S")
        # generate_forecasting_h5(self.pred_trajs, output_path=self.save_dir, filename=h5file_name, probabilities=self.pred_prbs)
        logging.info(f"\nGenerate submission file in {self.save_dir}/{h5file_name}")