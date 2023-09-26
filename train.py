import datetime
import re

import torch
import torch.optim as optim
import copy
import yaml
import os
import sys

from executor import MyExecutor
from init_model import init_model

curPath = os.path.abspath(os.path.dirname(__file__))
rootPath = os.path.split(curPath)[0]
sys.path.append(rootPath)
from tensorboardX import SummaryWriter
from wenet.dataset.dataset import Dataset
from wenet.utils.file_utils import read_symbol_table, read_domain_table
from torch.utils.data import DataLoader
from wenet.utils.scheduler import WarmupLR


def main():
    torch.manual_seed(777)
    config_path = "conf/train_conformer_moe.yaml"
    with open(config_path, 'r') as fin:
        configs = yaml.load(fin, Loader=yaml.FullLoader)

    data_type = "raw"
    data_root = "data/sc_km/"
    train_data = os.path.join(data_root, "train/data.list")
    cv_data = os.path.join(data_root, "dev/data.list")
    symbol_path = "data/dict/"
    model_dir = "exp/conformer_moe_e4"
    checkpoint_path = os.path.join(model_dir, '0.pt')
    load_mode = "moe"
    cmvn_path =  os.path.join(model_dir,"global_cmvn") if os.path.exists(os.path.join(model_dir,"global_cmvn")) else None
    
    exp_id = os.path.basename(model_dir)  # model_dir 的上一级
    tensorboard_path = os.path.join("tensorboard", exp_id)

    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    if not os.path.exists(tensorboard_path):
        os.makedirs(tensorboard_path)

    writer = SummaryWriter(tensorboard_path)

    # read vocab and accent
    symbol_table = read_symbol_table(symbol_path)
    domain_table = read_domain_table(symbol_path)

    vocab_size = len(symbol_table)
    domain_num = len(domain_table)

    if 'fbank_conf' in configs['dataset_conf']:
        input_dim = configs['dataset_conf']['fbank_conf']['num_mel_bins']
    else:
        input_dim = configs['dataset_conf']['mfcc_conf']['num_mel_bins']

    train_conf = configs['dataset_conf']
    cv_conf = copy.deepcopy(train_conf)
    cv_conf['speed_perturb'] = False
    cv_conf['spec_aug'] = False
    cv_conf['spec_sub'] = False
    cv_conf['spec_trim'] = False
    cv_conf['shuffle'] = False

    pin_memory = True
    num_workers = 1
    prefetch = 500

    train_dataset = Dataset(data_type, train_data, symbol_table, domain_table, train_conf)
    cv_dataset = Dataset(data_type,
                         cv_data,
                         symbol_table,
                         domain_table,
                         cv_conf,
                         partition=False)

    train_data_loader = DataLoader(train_dataset,
                                   batch_size=None,
                                   pin_memory=pin_memory,
                                   num_workers=num_workers,
                                   prefetch_factor=prefetch)
    cv_data_loader = DataLoader(cv_dataset,
                                batch_size=None,
                                pin_memory=pin_memory,
                                num_workers=num_workers,
                                prefetch_factor=prefetch)

    configs['input_dim'] = input_dim
    configs['output_dim'] = vocab_size
    configs['cmvn_file'] = cmvn_path
    configs['is_json_cmvn'] = True
    configs["domain_num"] = domain_num

    saved_config_path = os.path.join(model_dir, 'train.yaml')
    with open(saved_config_path, 'w') as fout:
        data = yaml.dump(configs)
        fout.write(data)    

    # init model
    model = init_model(configs)
    print(model)
    num_params = sum(p.numel() for p in model.parameters())
    print('the number of model params: {:,d}'.format(num_params))

    # 可以将带有注释的 Python 函数或类转换为 Torch 脚本。转换后的 Torch 脚本可以在不依赖 Python 解释器的环境中进行执行，从而提供了更高的性能和部署的灵活性。
    # script_model = torch.jit.script(model)
    # script_model.save(os.path.join(model_dir, 'init.zip'))

    executor = MyExecutor()

    # load checkpoint
    # start_epoch is checkpoint model name
    if checkpoint_path is not None:
        if load_mode == "moe":
            model.load_no_moe_checkpoint(checkpoint_path)
        else:
            checkpoint = torch.load(checkpoint_path)
            model.load_state_dict(checkpoint, strict=False)
        info_path = re.sub('.pt$', '.yaml', checkpoint_path)
        infos = {}
        if os.path.exists(info_path):
            with open(info_path, 'r') as fin:
                infos = yaml.load(fin, Loader=yaml.FullLoader)
        print(f"load checkpoint {checkpoint_path}")
    else:
        infos = {}

    # load checkpoint info for start_epoch, step, config ...
    start_epoch = infos.get('epoch', -1) + 1
    cv_loss = infos.get('cv_loss', 0.0)
    step = infos.get('step', -1)
    num_epochs = configs.get('max_epoch', 100)

    # Whether to use gpu
    use_cuda = torch.cuda.is_available()
    device = torch.device('cuda' if use_cuda else 'cpu')
    model = model.to(device)
    if configs['optim'] == 'adamw':
        optimizer = optim.AdamW(model.parameters(), **configs['optim_conf'])
    elif configs['optim'] == 'adam':
        optimizer = optim.Adam(model.parameters(), **configs['optim_conf'])
        # optimizer = optim.Adam(filter(lambda p: p.requires_grad, model_ft.parameters()), **configs['optim_conf'])
    else:
        raise ValueError("unknown optimizer: " + configs['optim'])
    
    scheduler = WarmupLR(optimizer, **configs['scheduler_conf'])

    # save init model
    if start_epoch == 0:
        save_path = os.path.join(model_dir, 'init.pt')
        state_dict = model.state_dict()
        torch.save(state_dict, save_path)
        info_path = re.sub('.pt$', '.yaml', save_path)
        if infos is None:
            infos = {}
        infos['save_time'] = datetime.datetime.now().strftime('%d/%m/%Y %H:%M:%S')
        with open(info_path, 'w') as fout:
            data = yaml.dump(infos)
            fout.write(data)

    executor.step = step
    scheduler.set_step(step)

    final_epoch = None

    for epoch in range(start_epoch, num_epochs):
        train_dataset.set_epoch(epoch)
        configs['epoch'] = epoch
        lr = optimizer.param_groups[0]['lr']
        print('Epoch {} TRAIN info lr {}'.format(epoch, lr))
        executor.train(model, optimizer, scheduler, train_data_loader, device,
                       writer, configs)

        total_loss, num_seen_utts = executor.cv(model, cv_data_loader, device,
                                                configs)

        cv_loss = total_loss / num_seen_utts

        print('Epoch {} CV info cv_loss {}'.format(epoch, cv_loss))
        
        infos = {
            'epoch': epoch, 'lr': lr, 'cv_loss': cv_loss, 'step': executor.step,
            'save_time': datetime.datetime.now().strftime('%d/%m/%Y %H:%M:%S')
        }

        writer.add_scalar('epoch/cv_loss', cv_loss, epoch)
        writer.add_scalar('epoch/lr', lr, epoch)
        # save model and infos
        with open("{}/{}.yaml".format(model_dir, epoch), 'w') as fout:
            data = yaml.dump(infos)
            fout.write(data)

        save_path = os.path.join(model_dir, '{}.pt'.format(epoch))
        state_dict = model.state_dict()
        torch.save(state_dict, save_path)
        info_path = re.sub('.pt$', '.yaml', save_path)
        if infos is None:
            infos = {}
        infos['save_time'] = datetime.datetime.now().strftime('%d/%m/%Y %H:%M:%S')
        with open(info_path, 'w') as fout:
            data = yaml.dump(infos)
            fout.write(data)

        final_epoch = epoch

    if final_epoch is not None:
        # save final model
        final_model_path = os.path.join(model_dir, 'final.pt')
        os.remove(final_model_path) if os.path.exists(final_model_path) else None
        os.symlink('{}.pt'.format(final_epoch), final_model_path)  # create a shortcut
        writer.close()

if __name__ == "__main__":
    main()
