import torch
import logging
import numpy as np
from tqdm import tqdm

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def eval_for_tail(eval_data, model, device, data, descending):
    hits = []
    ranks = []
    ent_rel_multi_t = data['entity_relation']['as_tail']
    for _ in range(10):  # need at most Hits@10
        hits.append([])

    for batch_idx, batch_data in enumerate(tqdm(eval_data)):
        eval_h = batch_data[0].to(device)
        eval_t = batch_data[1].to(device)
        eval_r = batch_data[2].to(device)
        _, pred = model(batch_h=eval_h, batch_r=eval_r)  # evaluate corruptions by replacing the object, i.e. tail entity

        # need to filter out the entities ranking above the target entity that form a
        # true (head, tail) entity pair in train/valid/test data
        for i in range(eval_h.size(0)):
            # get all tail entities that form triples with eval_h[i] as the head entity and eval_r[i] as the relation
            filter_t = ent_rel_multi_t[eval_h[i].item()][eval_r[i].item()]

            pred_value = pred[i][eval_t[i].item()].item()
            pred[i][filter_t] = 0.0
            pred[i][eval_t[i].item()] = pred_value

        _, index = torch.sort(pred, 1, descending=True)  # pred: (batch_size, ent_count)
        index = index.cpu().numpy()  # index: (batch_size)

        for i in range(eval_h.size(0)):
            # find the rank of the target entities
            rank = np.where(index[i] == eval_t[i].item())[0][0]

            # rank+1, since the rank starts with 1 not 0
            ranks.append(rank + 1)

            for hits_level in range(10):
                if rank <= hits_level:
                    hits[hits_level].append(1.0)
                else:
                    hits[hits_level].append(0.0)

    return hits, ranks

def eval_for_head(eval_data, model, device, data, descending, rel_cnt):
    hits = []
    ranks = []
    ent_rel_multi_h = data['entity_relation']['as_head']
    for _ in range(10):  # need at most Hits@10
        hits.append([])

    for batch_idx, batch_data in enumerate(tqdm(eval_data)):
        eval_h = batch_data[0].to(device)
        eval_t = batch_data[1].to(device)
        eval_r_inv = batch_data[2].to(device) + rel_cnt
        eval_r = batch_data[2].to(device)

        _, pred = model(batch_h=eval_t, batch_r=eval_r_inv)  # evaluate corruptions by replacing the object, i.e. tail entity

        # need to filter out the entities ranking above the target entity that form a
        # true (head, tail) entity pair in train/valid/test data
        for i in range(eval_t.size(0)):
            # get all tail entities that form triples with eval_h[i] as the head entity and eval_r[i] as the relation
            filter_h = ent_rel_multi_h[eval_t[i].item()][eval_r[i].item()]

            pred_value = pred[i][eval_h[i].item()].item()
            pred[i][filter_h] = 0.0
            pred[i][eval_h[i].item()] = pred_value

        _, index = torch.sort(pred, 1, descending=True)  # pred: (batch_size, ent_count)
        index = index.cpu().numpy()  # index: (batch_size)

        for i in range(eval_t.size(0)):
            # find the rank of the target entities
            rank = np.where(index[i] == eval_h[i].item())[0][0]

            # rank+1, since the rank starts with 1 not 0
            ranks.append(rank + 1)

            for hits_level in range(10):
                if rank <= hits_level:
                    hits[hits_level].append(1.0)
                else:
                    hits[hits_level].append(0.0)

    return hits, ranks

def eval_for_double(eval_data, model, device, data, descending, rel_cnt):
    hits_h, ranks_h = eval_for_head(eval_data, model, device, data, descending, rel_cnt)
    hits_t, ranks_t = eval_for_tail(eval_data, model, device, data, descending)
    hits = torch.cat((torch.tensor(hits_h), torch.tensor(hits_t)), dim=1)
    ranks = torch.cat((torch.tensor(ranks_t), torch.tensor(ranks_h)), dim=0)
    return hits.tolist(), ranks.tolist()

def output_eval_tail(results, data_name):
    hits = np.array(results[0])
    ranks = np.array(results[1])
    r_ranks = 1.0 / ranks  # compute reciprocal rank

    # print Hits@10, Hits@3, Hits@1, MR (mean rank), and MRR (mean reciprocal rank)
    print('For %s data: Hits@10=%.4f - Hits@3=%.4f - Hits@1=%.4f' % (data_name, hits[9].mean(), hits[2].mean(), hits[0].mean()))
    print('For %s data: MR=%.4f - MRR=%.4f' % (data_name, ranks.mean(), r_ranks.mean()))