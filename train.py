import logging
from tqdm import tqdm
from torch.utils.data import DataLoader
from utils import generateGraph


logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def train_without_label(dataset, model, optimizer, device, batch_size=None):
    data = DataLoader(dataset.data['train'], batch_size, drop_last=False, shuffle=True)
    full_loss = []
    model.train()
    rel_cnt = model.relation_cnt
    for batch_data in tqdm(data):
        h = batch_data[0].to(device)
        t = batch_data[1].to(device)
        r = batch_data[2].to(device)
        r_ = batch_data[2].to(device) + rel_cnt
        optimizer.zero_grad()
        loss_to_tail, _ = model(h, r, t)  # (h,r,t)
        loss_to_head, _ = model(t, r_, h)  # (t,r',h)
        # loss = loss.mean()
        loss = (loss_to_tail + loss_to_head)/2
        loss.backward()
        optimizer.step()
        full_loss.append(loss.item())
    return full_loss

#graphormer应用在知识图谱中
def train_graphormer(dataset, model, optimizer, device, batch_size=None):
    model.G = generateGraph(dataset.data['train'], model.relation_cnt)  # 模型将训练用的三元组以图的形式加载,该图为无向图
    data = DataLoader(dataset.data['train'], batch_size, drop_last=False, shuffle=True)
    full_loss = []
    model.train()
    rel_cnt = model.relation_cnt
    for batch_data in tqdm(data):
        h = batch_data[0].to(device)
        t = batch_data[1].to(device)
        r = batch_data[2].to(device)
        r_ = batch_data[2].to(device) + rel_cnt
        optimizer.zero_grad()
        loss_to_tail, _ = model(h, r, t)  # (h,r,t)
        loss_to_head, _ = model(t, r_, h)  # (t,r',h)
        # loss = loss.mean()
        loss = (loss_to_tail + loss_to_head)/2
        loss.backward()
        optimizer.step()
        full_loss.append(loss.item())
    return full_loss