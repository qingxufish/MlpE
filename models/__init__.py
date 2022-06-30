import torch
import logging
from torch.utils.data import DataLoader
from tqdm import tqdm
from .MlpE import MlpE

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

model_dict = {
    'MlpE': MlpE
}


def init_model(config, exp_class, args):
    # initialize model
    device = config.get('device')
    if config.get('model_name') in model_dict:
        model = model_dict[config.get('model_name')].init_model(config)
    else:
        raise ValueError('Model not support: ' + config.get('model_name'))
    logging.info(model)

    '''
    # For simplicity, use DataParallel wrapper to use multiple GPUs.
    if device == 'cuda' and torch.cuda.device_count() > 1:
        logging.info(f'{torch.cuda.device_count()} GPUs are available. Let\'s use them.')
        model = torch.nn.DataParallel(model)
    '''
    try:
        model = load_link(exp_class, model)
        model = model.to(device)
        logging.info(f'model loaded on {device}')
    except AttributeError:
        model = model.to(device)
        logging.info(f'model loaded on {device}')

    param_count = 0
    for p in model.parameters():
        param_count += p.view(-1).size()[0]
    logging.info(f'model param is {param_count}')

    return model, device


def load_link(exp_class, model):
    train_data = DataLoader(exp_class.dataset.data['double_triplets_train'], exp_class.train_conf.get('batch_size'), drop_last=False)
    for batch_data in tqdm(train_data):
        n_r = batch_data[1]
        r = batch_data[3]
        model.multi_infer.memorize_link(n_r, r)  # 记录所有共同出现的关系
    return model