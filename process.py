import time
import torch
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, mean_squared_error, r2_score, mean_absolute_error
from tqdm import tqdm
from loss import CE, Align, Reconstruct, Regress
from torch.optim.lr_scheduler import LambdaLR
from classification import fit_lr, get_rep_with_label
from torch.utils.tensorboard import SummaryWriter
import tensorflow as tf
import os
from joblib import Parallel, delayed
from model.TimeMAE import TimeMAE
from args import args
import re
# from memory_profiler import profile
# tensorflow metal 

# Top-level function (outside of any class)
# def compute_batch_static(model_state_dict, batch, device, alpha, beta):
#     model = YourModelClass(...)  # Re-initialize model
#     model.load_state_dict(model_state_dict)
#     model.to(device)
#     model.eval()

#     align = Align()
#     reconstruct = Reconstruct()

#     with torch.no_grad():
#         [rep_mask, rep_mask_prediction], [token_prediction_prob, tokens] = model.pretrain_forward(batch[0].to(device))
#         align_loss = align.compute(rep_mask, rep_mask_prediction)
#         reconstruct_loss, hits, ndcg = reconstruct.compute(token_prediction_prob, tokens)
#         loss = alpha * align_loss + beta * reconstruct_loss
#     return loss.item(), align_loss.item(), reconstruct_loss.item(), hits.item(), ndcg

# def compute_batch(model, batch, device, align, reconstruct, alpha, beta):
#         batch = [x.to(device, non_blocking=True) for x in batch]
#         [rep_mask, rep_mask_prediction], [token_prediction_prob, tokens] = model.pretrain_forward(batch[0])
#         align_loss = align.compute(rep_mask, rep_mask_prediction)
#         reconstruct_loss, hits, ndcg = reconstruct.compute(token_prediction_prob, tokens)
#         total_loss = alpha * align_loss + beta * reconstruct_loss
#         return (total_loss.item(), align_loss.item(), reconstruct_loss.item(), hits.item(), ndcg)


def process_sub_batch(model, sub_batch, device, alpha, beta):
    rep_mask, rep_mask_prediction = model.pretrain_forward(sub_batch[0])[0]
    token_prediction_prob, tokens = model.pretrain_forward(sub_batch[0])[1]
    align_loss = Align().compute(rep_mask, rep_mask_prediction)
    reconstruct_loss, hits, NDCG = Reconstruct().compute(token_prediction_prob, tokens)
    loss = alpha * align_loss + beta * reconstruct_loss
    return loss.item(), align_loss.item(), reconstruct_loss.item(), hits.item(), NDCG

class Trainer_Regress():
    def __init__(self, args, model, train_loader, train_linear_loader, test_loader, verbose=False):
        self.args = args
        self.verbose = verbose
        self.device = args.device
        self.print_process(self.device)
        self.model = model.to(torch.device(self.device))
        self.epoch = 0
        # self.model = model.cuda()
        # print('model cuda')

        self.train_loader = train_loader
        self.train_linear_loader = train_linear_loader
        self.test_loader = test_loader

        self.lr = args.lr
        self.lr_decay_rate = args.lr_decay_rate
        self.lr_decay_steps = args.eval_per_steps #args.lr_decay_steps
        self.weight_decay = args.weight_decay
        self.regression_loss = Regress(self.model)
        self.alpha = args.alpha
        self.beta = args.beta

        self.test_mse = torch.nn.MSELoss()
        self.num_epoch = args.num_epoch
        self.num_epoch_pretrain = args.num_epoch_pretrain
        self.eval_per_steps = args.eval_per_steps
        self.save_path = args.save_path
        self.save_freq = args.save_freq
        # if self.num_epoch:
        #     self.result_file = open(self.save_path + '/result.txt', 'w')
        #     self.result_file.close()
        
        self.step = 0
        self.best_metric = -1e9
        self.best_metric_dopa = -1e9
        self.best_metric_sero = -1e9
        self.metric = 'r2'
        log_dir = os.path.join(self.save_path, 'runs')
        os.makedirs(log_dir, exist_ok=True)
        self.writer = SummaryWriter(log_dir)
       
    # @profile
    def pretrain(self):
        start = time.perf_counter()
        print('pretraining')
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.args.lr)
        # model = TimeMAE(args)
        # device = 'cpu'
        # alpha = 5.0
        # beta = 1.0
        eval_mse = 0
        align = Align()
        reconstruct = Reconstruct()
        self.model.copy_weight()
        if self.num_epoch_pretrain:
            result_file = open(self.save_path + '/pretrain_result.txt', 'w')
            result_file.close()
            result_file = open(self.save_path + '/linear_result.txt', 'w')
            result_file.close()
        for epoch in range(self.num_epoch_pretrain):
            self.model.train()
            tqdm_dataloader = tqdm(self.train_loader)
            loss_sum = 0
            loss_mse = 0
            loss_ce = 0
            hits_sum = 0
            NDCG_sum = 0
            for idx, batch in enumerate(tqdm_dataloader):
                batch = [x.to(self.device) for x in batch]
                self.optimizer.zero_grad()
                [rep_mask, rep_mask_prediction], [token_prediction_prob, tokens] = self.model.pretrain_forward(batch[0])
                align_loss = align.compute(rep_mask, rep_mask_prediction)
                loss_mse += align_loss.item()
                reconstruct_loss, hits, NDCG = reconstruct.compute(token_prediction_prob, tokens)
                loss_ce += reconstruct_loss.item()
                hits_sum += hits.item()
                NDCG_sum += NDCG
                loss = self.alpha * align_loss + self.beta * reconstruct_loss
                loss.backward()
                self.optimizer.step()
                self.model.momentum_update()
                loss_sum += loss.item()
            avg_loss = loss_sum / (idx + 1)
            avg_mse = loss_mse / (idx + 1)
            avg_ce = loss_ce / (idx + 1)
            avg_ndcg = NDCG_sum / (idx + 1)        
            print('pretrain epoch{0}, loss{1}, mse{2}, ce{3}, hits{4}, ndcg{5}'.format(epoch + 1, loss_sum / (idx + 1),
                                                                                        loss_mse / (idx + 1),
                                                                                        loss_ce / (idx + 1), hits_sum,
                                                                                        NDCG_sum / (idx + 1)))

            with open(self.save_path + '/pretrain_result.txt', 'a+') as result_file:
                print(f'pretrain epoch{epoch + 1}, loss{avg_loss}, mse{avg_mse}, ce{avg_ce}, hits{hits_sum}, ndcg{avg_ndcg}', file=result_file)
                
            self.writer.add_scalar("Pretrain/Loss", float(avg_loss), int(epoch))
            self.writer.add_scalar("Pretrain/MSE", avg_mse, epoch)
            self.writer.add_scalar("Pretrain/CE_Loss", avg_ce, epoch)
            self.writer.add_scalar("Pretrain/Hits", hits_sum, epoch)
            self.writer.add_scalar("Pretrain/NDCG", avg_ndcg, epoch)
            self.writer.add_scalar("Pretrain/rainigLearningRate", self.optimizer.param_groups[0]['lr'], epoch)
            
            if (self.save_freq is not None) and (epoch % self.save_freq == 0) and \
					(epoch > 0):            
                self.model.eval()
                train_rep, train_label = get_rep_with_label(self.model, self.train_linear_loader)
                test_rep, test_label = get_rep_with_label(self.model, self.test_loader)        
                reg = LinearRegression().fit(train_rep, train_label)
                preds = reg.predict(test_rep)
                r2 = r2_score(test_label, preds)
                mse = mean_squared_error(test_label, preds)
                print('r2:{0}, mse:{1}'.format(r2, mse))
                result_file = open(self.save_path + '/regression_linear_result.txt', 'a+')
                print('epoch{0}, r2{1}, mse{2}'.format(0, r2, mse), file=result_file)
                result_file.close()
                torch.save(self.model.state_dict(), self.save_path + '/pretrain_model_checkpoint_' +str(epoch).zfill(3) +'.pkl')
            
        print(f"[Timing] eval_model took {time.perf_counter() - start:.2f} seconds")


    # @profile
    def finetune(self):
        start = time.perf_counter()
        print('finetune')
        if self.args.resume:
            if self.args.load_pretrained_model:
                print('load pretrained model')
                state_dict = torch.load(self.save_path + '/pretrain_model.pkl', map_location=self.device)
                try:
                    self.model.load_state_dict(state_dict)
                except:
                    model_state_dict = self.model.state_dict()
                    for pretrain, random_intial in zip(state_dict, model_state_dict):
                        assert pretrain == random_intial
                        if pretrain in ['input_projection.weight', 'input_projection.bias', 'predict_head.weight',
                                        'predict_head.bias', 'position.pe.weight']:
                            state_dict[pretrain] = model_state_dict[pretrain]
                    self.model.load_state_dict(state_dict)
            else:
                resume_checkpoint = get_latest_checkpoint_from_folder(self.save_path)
                print(f"Resuming from checkpoint: {resume_checkpoint}")
                checkpoint = torch.load(os.path.join(self.save_path, resume_checkpoint), map_location=self.device)
                self.model.state_dict(checkpoint['model_state_dict'])
                self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.args.lr)
                self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                self.step = checkpoint.get('step', 0)
                self.best_metric = checkpoint.get('best_metric', -1e9)
                self.epoch = checkpoint['epoch'] + 1
                print(f"Training epochs: {self.epoch} to {self.epoch + self.args.num_epoch}")
        
        self.model.eval()
        train_rep, train_label = get_rep_with_label(self.model, self.train_linear_loader)
        test_rep, test_label = get_rep_with_label(self.model, self.test_loader)        
        reg = LinearRegression().fit(train_rep, train_label)
        preds = reg.predict(test_rep)
        r2 = r2_score(test_label, preds)
        mse = mean_squared_error(test_label, preds)
        print('r2:{0}, mse:{1}'.format(r2, mse))
        
        with open(self.save_path + '/regression_result.txt', 'a+') as result_file:
            print(f'epoch{0}, r2{r2}, mse{mse}', file=result_file)

        self.model.linear_proba = False
        # self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.args.lr) #original
        # self.scheduler = LambdaLR(self.optimizer, lr_lambda=lambda epoch: self.lr_decay ** epoch) #Original
        # self.scheduler = LambdaLR(self.optimizer, lr_lambda=lambda step: self.lr_decay ** step, verbose=self.verbose)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        self.scheduler = LambdaLR(self.optimizer, lr_lambda=lambda step: self.lr_decay_rate ** (step/self.lr_decay_steps))
       
        for epoch in range(self.epoch, self.epoch + self.num_epoch):
            loss_epoch, time_cost = self._train_one_epoch()
            # self.scheduler.step()
            metric = self.eval_model()
            if self.args.num_targets == 1:  
                self.writer.add_scalar("test/R2", metric['r2'], epoch)
                self.writer.add_scalar("test/MSE", metric['mse'], epoch)
                self.writer.add_scalar("test/MAE", metric['mae'], epoch)
            else:
                self.writer.add_scalar("test/R2", metric['r2'], epoch)
                self.writer.add_scalar("test/MSE", metric['mse'], epoch) 
                self.writer.add_scalar("test/MSE_dopamine", metric['mse_dopa'], epoch)
                self.writer.add_scalar("test/MAE_dopamine", metric['mae_dopa'], epoch)
                self.writer.add_scalar("test/R2_dopamine", metric['r2_dopa'], epoch)
                self.writer.add_scalar("test/MSE_serotonin", metric['mse_sero'], epoch)
                self.writer.add_scalar("test/MAE_serotonin", metric['mae_sero'], epoch)
                self.writer.add_scalar("test/R2_serotonin", metric['r2_sero'], epoch)
            self.writer.add_scalar("train/Loss", loss_epoch, epoch)    
            self.writer.add_scalar("LearningRate", self.optimizer.param_groups[0]['lr'], epoch)
            if (self.save_freq is not None) and (epoch % self.save_freq == 0) and \
					(epoch > 0):
                filename = "/finetune_model_checkpoint_"+str(epoch).zfill(3)+'.pkl'
                self.save_state(filename)
                
            with open(self.save_path + '/result.txt', 'a+') as result_file:
                self.print_process(f'Finetune epoch:{epoch + 1},loss:{loss_epoch},training_time:{time_cost}')
                print(f'Finetune train epoch:{epoch + 1},loss:{loss_epoch},training_time:{time_cost}', file=result_file)
            
        self.writer.close()    
        self.print_process(self.best_metric)
        print(f"[Timing] finetune took {time.perf_counter() - start:.2f} seconds")
        return self.best_metric
    
    # @profile
    def _train_one_epoch(self):
        t0 = time.perf_counter()
        self.model.train()
        tqdm_dataloader = tqdm(self.train_linear_loader) if self.verbose else self.train_linear_loader

        loss_sum = 0
        for idx, batch in enumerate(tqdm_dataloader):
            batch = [x.to(self.device) for x in batch]

            self.optimizer.zero_grad()
            loss = self.regression_loss.compute(batch)
            loss_sum += loss.item()

            loss.backward()
            # torch.nn.utils.clip_grad_norm_(self.model.parameters(), 5)
            self.optimizer.step()
            self.scheduler.step()
            self.step += 1
        # if self.step % self.eval_per_steps == 0:
        metric = self.eval_model()
        self.print_process(metric)
        
        # with open(self.save_path + '/result.txt', 'a+') as result_file:
        #     print(f'step{self.step}', file=result_file)
        #     print(metric, file=result_file)

        if self.args.num_targets == 1: 
            if metric[self.metric] >= self.best_metric:
                torch.save(self.model.state_dict(), self.save_path + '/model.pkl')
                with open(self.save_path + '/result.txt', 'a+') as result_file:
                    print('saving model of step{0}'.format(self.step), file=result_file)
                self.best_metric = metric[self.metric]  
        else:
            if metric[self.metric] >= self.best_metric:
                torch.save(self.model.state_dict(), self.save_path + '/model.pkl')
                with open(self.save_path + '/result.txt', 'a+') as result_file:
                    print('saving model of step{0}'.format(self.step), file=result_file)
                self.best_metric = metric[self.metric]  

            if metric['r2_dopa'] >= self.best_metric_dopa:
                torch.save(self.model.state_dict(), self.save_path + '/dopa_model.pkl')
                with open(self.save_path + '/dopa_result.txt', 'a+') as result_file:
                    print('saving model of step{0}'.format(self.step), file=result_file)
                self.best_metric_dopa = metric['r2_dopa']

            if metric['r2_sero'] >= self.best_metric_sero:
                torch.save(self.model.state_dict(), self.save_path + '/sero_model.pkl')
                with open(self.save_path + '/sero_result.txt', 'a+') as result_file:
                    print('saving model of step{0}'.format(self.step), file=result_file)
                self.best_metric_sero = metric['r2_sero'] 
           
        # self.model.train()
        self.epoch += 1
        print(f"[Timing] train_one_epoch took {time.perf_counter() - t0:.2f} seconds")
        return loss_sum / (idx + 1), time.perf_counter() - t0
    
    # @profile
    def eval_model(self):
        self.model.eval()
        start = time.perf_counter()
        tqdm_data_loader = tqdm(self.test_loader) if self.verbose else self.test_loader
        metrics = {'mse': 0, 'r2': 0}
        pred = []
        label = []
        test_loss = 0

        with torch.no_grad():
            for idx, batch in enumerate(tqdm_data_loader):
                batch = [x.to(self.device) for x in batch]
                ret = self.compute_metrics(batch)
                if len(ret) == 2:
                    pred_b, label_b = ret
                    pred += pred_b
                    label += label_b
                else:
                    pred_b, label_b, test_loss_b = ret
                    pred += pred_b
                    label += label_b
                    test_loss += test_loss_b.cpu().item()
        if self.args.num_targets == 1: 
            metrics['mse'] = mean_squared_error(label, pred)
            metrics['r2'] = r2_score(label, pred)
            metrics['test_loss'] = test_loss / (idx + 1)
            metrics['mae'] = mean_absolute_error(label, pred)
        else: 
            pred_np = np.array(pred)
            label_np = np.array(label)
            metrics['mse'] = mean_squared_error(label_np, pred_np)
            metrics['r2'] = r2_score(label_np, pred_np)
            metrics['mse_dopa'] = mean_squared_error(label_np[:, 0], pred_np[:, 0])
            metrics['mse_sero'] = mean_squared_error(label_np[:, 1], pred_np[:, 1])
            metrics['r2_dopa'] = r2_score(label_np[:, 0], pred_np[:, 0])
            metrics['r2_sero'] = r2_score(label_np[:, 1], pred_np[:, 1])
            metrics['mae_dopa'] = mean_absolute_error(label_np[:, 0], pred_np[:, 0])
            metrics['mae_sero'] = mean_absolute_error(label_np[:, 1], pred_np[:, 1])
            metrics['test_loss'] = test_loss / (idx + 1)
        
        print(f"[Timing] eval_model took {time.perf_counter() - start:.2f} seconds")

        # In Trainer_Regress.eval_model(self) just before `return metrics`
        epoch_dir = os.path.join(self.save_path, "preds")
        os.makedirs(epoch_dir, exist_ok=True)
        np.save(os.path.join(epoch_dir, f"pred_epoch_{self.epoch:03d}.npy"), np.array(pred))
        np.save(os.path.join(epoch_dir, f"label_epoch_{self.epoch:03d}.npy"), np.array(label))        
        return metrics
    
    def compute_metrics(self, batch):
        if len(batch) == 2:
            seqs, label = batch
            scores = self.model(seqs)
        else:
            seqs1, seqs2, label = batch
            scores = self.model((seqs1, seqs2))

        # pred = scores.squeeze()
        # label = label.view_as(pred)
        pred = scores
        if pred.shape != label.shape:
            raise ValueError(f"Shape mismatch: pred {pred.shape}, label {label.shape}")

        test_loss = self.test_mse(pred, label)
        return pred.tolist(), label.tolist(), test_loss


    def print_process(self, *x):
        if self.verbose:
            print(*x)

    def save_state(self, filename):
        """Save all the model parameters to the given file."""
        checkpoint = {
        'epoch': self.epoch,
        'model_state_dict': self.model.state_dict(),
        'optimizer_state_dict': self.optimizer.state_dict(),
        'best_metric': self.best_metric,
        'step': self.step
        }
        
        torch.save(checkpoint, self.save_path + filename)


def get_latest_checkpoint_from_folder(folder_path):
    # List all files in the folder
    all_files = os.listdir(folder_path)
    
    # Filter files that start with 'checkpoint' and end with '.pkl'
    checkpoint_files = [f for f in all_files if f.startswith("finetune") and f.endswith(".pkl")]
    
    # Extract numbers and find the file with the largest number
    latest_file = max(checkpoint_files, key=lambda name: int(re.search(r'(\d+)', name).group()))
    return latest_file