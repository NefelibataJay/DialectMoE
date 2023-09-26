import torch
from torch.nn.utils import clip_grad_norm_
from contextlib import nullcontext

class MyExecutor:
    def __init__(self):
        self.step = 0

    def train(self, model, optimizer, scheduler, data_loader, device, writer, configs):
        model.train()
        clip = configs.get('grad_clip', 50.0)
        accum_grad = configs.get('accum_grad', 1)
        num_seen_utts = 0
        epoch = configs.get('epoch', 0)
        logger_interval = configs.get('logger_interval', 100)
        print('using accumulate grad, new batch size is {} times larger than before'.format(accum_grad))

        model_context = nullcontext
        
        for batch_idx, batch in enumerate(data_loader):
            key, feats, target, feats_lengths, target_lengths, domain_id = batch
            feats = feats.to(device)  # (batch_size, max_len, feat_dim)
            target = target.to(device)  # (batch_size, max_len)
            feats_lengths = feats_lengths.to(device)  # (batch_size)
            target_lengths = target_lengths.to(device)  # (batch_size)
            domain_id = domain_id.to(device)  # (batch_size)
            num_utts = target_lengths.size(0)

            if num_utts == 0:
                continue

            # context = nullcontext
            # important
            loss_dict = model(feats, feats_lengths, target, target_lengths, domain_id)
            loss = loss_dict['loss'] / accum_grad
            loss.backward()

            num_seen_utts += num_utts
            if batch_idx % accum_grad == 0:
                writer.add_scalar('train_loss', loss, self.step)
                grad_norm = clip_grad_norm_(model.parameters(), clip)
                if torch.isfinite(grad_norm):
                    # 检查梯度的范数是否出现非有限数 (如NaN或Inf)
                    # 与输入张量形状相同，其中每个元素的值为True或False，
                    optimizer.step()
                else:
                    print('grad norm is {}, skip update'.format(grad_norm))
                optimizer.zero_grad()
                scheduler.step()
                self.step += 1

            if batch_idx % logger_interval == 0:
                lr = optimizer.param_groups[0]['lr']
                log_str = 'TRAIN Batch {}/{} loss {:.6f} '.format(
                    epoch, batch_idx,
                    loss.item() * accum_grad)
                for name, value in loss_dict.items():
                    if name != 'loss' and value is not None:
                        log_str += '{} {:.6f} '.format(name, value.item())
                log_str += 'lr {:.8f}'.format(lr)
                print(log_str)

    def cv(self, model, data_loader, device, configs):
        model.eval()
        num_seen_utts = 1
        total_loss = 0.0
        with torch.no_grad():
            for batch_idx, batch in enumerate(data_loader):
                key, feats, target, feats_lengths, target_lengths, domain_id = batch
                feats = feats.to(device)
                target = target.to(device)
                feats_lengths = feats_lengths.to(device)
                target_lengths = target_lengths.to(device)
                domain_id = domain_id.to(device)

                num_utts = target_lengths.size(0)
                if num_utts == 0:
                    continue
                # important
                loss_dict = model(feats, feats_lengths, target, target_lengths, domain_id)
                loss = loss_dict['loss']

                if torch.isfinite(loss):
                    num_seen_utts += num_utts
                    total_loss += loss.item() * num_utts

        return total_loss, num_seen_utts
