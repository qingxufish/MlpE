import os
import json
import torch
import logging
import numpy as np
from tqdm import tqdm
import codecs
import networkx as nx

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


def construct_dict(dir_path):
    """
    construct the entity2id, relation2id,train2id, valid2id, test2id and save them in txt format
    :param dir_path: data directory path
    :return:
    """
    ent2id, rel2id = dict(), dict()

    # index entities / relations in the occurence order in train, valid and test set
    train_path, valid_path, test_path = os.path.join(dir_path, 'train.txt'), os.path.join(dir_path, 'valid.txt'), \
                                        os.path.join(dir_path, 'test.txt')
    # save entity2id and relation2id
    for path in [train_path, valid_path, test_path]:
        with open(path, 'r', encoding='utf-8') as f:
            for line in f:
                h, r, t = line.split('\t')
                t = t[:-1]  # remove \n
                if h not in ent2id:
                    ent2id[h] = len(ent2id)
                if t not in ent2id:
                    ent2id[t] = len(ent2id)
                if r not in rel2id:
                    rel2id[r] = len(rel2id)

    # arrange the items in id order
    ent2id, rel2id = dict(sorted(ent2id.items(), key=lambda x: x[1])), dict(sorted(rel2id.items(), key=lambda x: x[1]))
    with codecs.open(os.path.join(dir_path, 'entity2id.txt'), 'w', encoding='utf-8') as f:
        for ind in ent2id.keys():
            f.write(ind + "\t" + str(ent2id[ind]) + '\n')

    with codecs.open(os.path.join(dir_path, 'relation2id.txt'), 'w', encoding='utf-8') as f:
        for ind in rel2id.keys():
            f.write(ind + "\t" + str(rel2id[ind]) + '\n')

    # save train2id, vaild2id and test2id
    data_name = ['train2id.txt', 'valid2id.txt', 'test2id.txt']
    for ind, path in enumerate([train_path, valid_path, test_path]):
        with open(path, 'r', encoding='utf-8') as f:
            for line in tqdm(f):
                h, r, t = line.split('\t')
                t = t[:-1]  # remove \n
                with codecs.open(os.path.join(dir_path, data_name[ind]), 'a', encoding='utf-8') as save_f:
                    save_f.write(str(ent2id[h]) + ' ' + str(ent2id[t]) + ' ' + str(rel2id[r]) + '\n')

    return True


def read_data(set_flag):
    """
    read data from file
    :param set_flag: train / valid / test set flag
    :return:
    """
    assert set_flag in [
        'train', 'valid', 'test',
        ['train', 'valid'], ['train', 'valid', 'test']
    ]
    cfg = utils.get_global_config()
    dir_p = join(cfg.dataset_dir, cfg.dataset)
    ent2id, rel2id = construct_dict(dir_p)

    # read the file
    if set_flag in ['train', 'valid', 'test']:
        path = join(dir_p, '{}.txt'.format(set_flag))
        file = open(path, 'r', encoding='utf-8')
    elif set_flag == ['train', 'valid']:
        path1 = join(dir_p, 'train.txt')
        path2 = join(dir_p, 'valid.txt')
        file1 = open(path1, 'r', encoding='utf-8')
        file2 = open(path2, 'r', encoding='utf-8')
        file = chain(file1, file2)
    elif set_flag == ['train', 'valid', 'test']:
        path1 = join(dir_p, 'train.txt')
        path2 = join(dir_p, 'valid.txt')
        path3 = join(dir_p, 'test.txt')
        file1 = open(path1, 'r', encoding='utf-8')
        file2 = open(path2, 'r', encoding='utf-8')
        file3 = open(path3, 'r', encoding='utf-8')
        file = chain(file1, file2, file3)
    else:
        raise NotImplementedError

    src_list = []
    dst_list = []
    rel_list = []
    pos_tails = defaultdict(set)
    pos_heads = defaultdict(set)
    pos_rels = defaultdict(set)

    for i, line in enumerate(file):
        h, r, t = line.strip().split('\t')
        h, r, t = ent2id[h], rel2id[r], ent2id[t]
        src_list.append(h)
        dst_list.append(t)
        rel_list.append(r)

        # format data in query-answer form
        # (h, r, ?) -> t, (?, r, t) -> h
        pos_tails[(h, r)].add(t)
        pos_heads[(r, t)].add(h)
        pos_rels[(h, t)].add(r)  # edge relation
        pos_rels[(t, h)].add(r + len(rel2id))  # inverse relations

    output_dict = {
        'src_list': src_list,
        'dst_list': dst_list,
        'rel_list': rel_list,
        'pos_tails': pos_tails,
        'pos_heads': pos_heads,
        'pos_rels': pos_rels
    }

    return output_dict


def load_json_config(config_path, args):
    logging.info(' Loading configuration '.center(100, '-'))
    if not os.path.exists(config_path):
        logging.warning(f'File {config_path} does not exist, empty list is returned.')
    with open(config_path, 'r') as f:
        config = json.load(f)
    if config['GPU']:
        config['device'] = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    else:
        config['device'] = torch.device('cpu')
    return config


def load_triples(file_path):
    tuples = []
    if not os.path.exists(file_path):
        logging.warning(f'File {file_path} does not exist, empty list is returned.')
        raise IOError("No such file")
    else:
        with open(file_path, 'r') as f:
            data = f.readlines()
            logging.info('%d triples loaded from %s.' % (len(data)-1, file_path))
            for line in data[1:]:
                record = line.strip().split(' ')
                tuples.append(tuple(map(int, record)))
    return tuples


def load_double_triples(file_path):
    tuples = []
    if not os.path.exists(file_path):
        logging.warning(f'File {file_path} does not exist, empty list is returned.')
        raise IOError("No such file")
    else:
        with open(file_path, 'r') as f:
            data = f.readlines()
            logging.info('%d triples loaded from %s.' % (len(data)-1, file_path))
            for line in data[1:]:
                record = line.strip().split('\t')
                tuples.append(tuple(map(int, record)))
    return tuples


def load_infer_triples(file_path):
    tuples = []
    if not os.path.exists(file_path):
        logging.warning(f'File {file_path} does not exist, empty list is returned.')
    else:
        with open(file_path, 'r') as f:
            data = f.readlines()
            logging.info('%d triples loaded from %s.' % (len(data)-1, file_path))
            for line in data[1:]:
                record = line.strip().split('\t')
                tuples.append(tuple(map(int, record)))

        triplets_list = list()
        temp_list = list()
        ind = 0
        for infer_triplet_ind, infer_triplet in enumerate(tuples):
            if tuples[ind][5] == infer_triplet[5]:
                temp_list.append(infer_triplet[:5])  # insert
            else:
                triplets_list.append(temp_list)
                ind = infer_triplet_ind
                temp_list = [infer_triplet[:5]]  # empty temp_list
    return triplets_list


def load_ids(file_path):
    ids = []
    if not os.path.exists(file_path):
        logging.warning(f'File {file_path} does not exist, empty list is returned.')
    else:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = f.readlines()
            logging.info('%d of entities/relations loaded from %s.' % (len(data)-1, file_path))
            for line in data[1:]:
                record = line.strip()
                try:
                    id = record.split('\t')[1]
                except IndexError:
                    id = record.split(' ')[-1]
                ids.append(int(id))
    return ids


def calc_goal_distribute(batch_h, batch_r, batch_t, goal_data, num_entities, num_relations):
    head_set = batch_h
    rel_set = batch_r
    tail_set = batch_t

    goal_dis = np.empty([rel_set.shape[0], num_entities])
    for triplet_id in range(int(rel_set.shape[0]/2)):
        head_id = head_set[triplet_id]
        rel_id = rel_set[triplet_id]
        tail_id = tail_set[triplet_id]

        head_rel_dis = torch.tensor(goal_data[head_id][rel_id])
        rel_tail_dis = torch.tensor(goal_data[tail_id][rel_id+num_relations])

        s2t_distribute = torch.zeros(num_entities)
        t2s_distribute = torch.zeros(num_entities)

        s2t_distribute = s2t_distribute.index_fill(0, head_rel_dis.to(torch.long), 1)
        t2s_distribute = t2s_distribute.index_fill(0, rel_tail_dis.to(torch.long), 1)

        goal_dis[triplet_id, :] = s2t_distribute.tolist()
        goal_dis[triplet_id+int(rel_set.shape[0]/2), :] = t2s_distribute.tolist()

    return goal_dis


def generate_goal_array(train_triplets, num_entities, num_relation):
    head_relation_triplets = torch.tensor(train_triplets[:, 0::2])
    relation_tail_triplets = torch.tensor(train_triplets[:, -1:-3:-1].copy())
    train_triplets = torch.tensor(train_triplets)

    goal_array = np.empty([num_entities, num_relation*2], dtype=list)  # 构建目标索引列表
    ind = 0
    for triplet in train_triplets:
        triplet = triplet.tolist()
        head_relation = torch.tensor(triplet[0::2])
        relation_tail = torch.tensor(triplet[-1:-3:-1].copy())
        triplet = torch.tensor(triplet)

        tail_index = torch.sum(head_relation_triplets == head_relation, dim=1)
        tail_index = torch.nonzero(tail_index == 2).squeeze()
        tail_index = train_triplets[tail_index, 1].view(-1).tolist()

        head_index = torch.sum(relation_tail_triplets == relation_tail, dim=1)
        head_index = torch.nonzero(head_index == 2).squeeze()
        head_index = train_triplets[head_index, 0].view(-1).tolist()

        goal_array[int(triplet[0])][int(triplet[2])] = tail_index
        goal_array[int(triplet[1])][int(triplet[2])+num_relation] = head_index
        ind = ind + 1
        print(ind)

    return goal_array

def generateGraph(trainDataset, relCount):
    G = nx.Graph()
    for single_edge in trainDataset:
        G.add_edge(single_edge[0], single_edge[1], edge_name=single_edge[2])
        G.add_edge(single_edge[1], single_edge[0], edge_name=single_edge[2]+relCount)
    return G