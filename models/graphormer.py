import matplotlib.pyplot as plt
import networkx as nx
import torch
from torch.nn.parameter import Parameter
from .BaseModel import BaseModel
from copy import deepcopy


class graphormer(BaseModel):
    def __init__(self, config):
        super(graphormer, self).__init__(config)
        self.device = config.get('device')
        self.entity_cnt = config.get('entity_cnt')
        self.relation_cnt = config.get('relation_cnt')
        kwargs = config.get('model_hyper_params')

        self.entity_dim = kwargs.get('entity_dim')  # 由于将关系嵌入的地位与实体嵌入等同，所以两者嵌入维度应当相同
        self.relation_dim = kwargs.get('relation_dim')

        self.E = torch.nn.Embedding(self.entity_cnt, self.entity_dim)
        self.R = torch.nn.Embedding(self.relation_cnt * 2, self.relation_dim)
        self.G = None #将三元组数据转化为图数据方便生成子图
        self.radius = kwargs.get('radius') # 生成子图的阶数

        path_kwargs = config.get('pathEmbParams')
        self.nhead = path_kwargs.get('head')
        self.layer_num = path_kwargs.get('layer_num')
        encoder_layer = torch.nn.TransformerEncoderLayer(d_model=self.entity_dim, nhead=self.nhead)
        self.pathEmbLayer = torch.nn.TransformerEncoder(encoder_layer, num_layers=self.layer_num)

        output_kwargs = config.get('outputParams')
        self.out_nhead = output_kwargs.get('head')
        self.out_layer_num = output_kwargs.get('layer_num')
        out_encoder_layer = torch.nn.TransformerEncoderLayer(d_model=self.entity_dim, nhead=self.out_nhead)
        self.outputLayer = torch.nn.TransformerEncoder(out_encoder_layer, num_layers=self.out_layer_num)

        self.input_drop = torch.nn.Dropout(kwargs.get('input_dropout'))
        self.feature_drop = torch.nn.Dropout(kwargs.get('feature_map_dropout'))
        self.hidden_drop = torch.nn.Dropout(kwargs.get('hidden_dropout'))

        self.bn0 = torch.nn.BatchNorm1d(1)  # batch normalization over a 4D input
        self.bn2 = torch.nn.BatchNorm1d(1)
        # self.register_parameter('b', Parameter(torch.zeros(self.entity_cnt)))
        self.register_parameter('c', Parameter(torch.ones(1)))

        self.encoder = torch.nn.Linear(self.entity_dim+self.relation_dim, self.entity_dim * self.relation_dim)
        self.activate = torch.nn.LeakyReLU(0.1)
        self.decoder = torch.nn.Linear(self.entity_dim * self.relation_dim, self.entity_dim)
        self.loss = ConvELoss(self.device, kwargs.get('label_smoothing'), self.entity_cnt)
        self.init()

    def init(self):
        torch.nn.init.xavier_normal_(self.E.weight.data)
        torch.nn.init.xavier_normal_(self.R.weight.data)

    @staticmethod
    def unitized(batch_vector):  # size:batch,length
        scalar = torch.norm(batch_vector, p=2, dim=1).view(-1, 1)
        scalar = scalar.expand_as(batch_vector)
        unitized_vector = batch_vector / scalar
        return unitized_vector

    def genInputSeqence(self,  centerNode):
        node = centerNode.tolist()[0]
        k_subGraph = nx.ego_graph(self.G, n=node, radius=self.radius)
        structTable = self.genStructTable(k_subGraph, node)
        structEmbTable = torch.tensor([]).to(self.device)
        for structItem in structTable:
            itemEmb = torch.tensor([]).to(self.device)
            structEmb = torch.tensor([]).to(self.device)
            for eleInd, ele in enumerate(structItem):
                if eleInd%2 == 0 :  # 偶数位置为节点，奇数位置为边
                    itemEmb = torch.cat([itemEmb, self.E(ele).view(-1,self.entity_dim)], dim=0)
                else:
                    itemEmb = torch.cat([itemEmb, self.R(ele).view(-1, self.entity_dim)], dim=0)
                structEmb = torch.cat([structEmb,
                                       self.pathEmbLayer(torch.tensor(itemEmb).to(self.device))[-1,:].view(-1, self.entity_dim)],
                                      dim=0).view(-1, self.entity_dim) # 得到该位置的结构嵌入
            structEmbTable = torch.cat([structEmbTable, itemEmb + structEmb], dim=0)
        return structEmbTable

    def genStructTable(self, k_subGraph, cNode):
        nodes = k_subGraph.nodes()

        structTable = []
        for neighbourNode in nodes:
            if neighbourNode == cNode:
                continue
            else:
                tempPath = nx.shortest_path(k_subGraph, cNode, neighbourNode)
                edgeList = []
                for nodeInd, edgeLeftNode in enumerate(tempPath):
                    if len(tempPath)-1 > nodeInd:
                        edgeRightNode = tempPath[nodeInd+1]
                        try:
                            edgeName = nx.get_edge_attributes(k_subGraph, 'edge_name')[(edgeLeftNode, edgeRightNode)]
                        except KeyError:
                            edgeName = nx.get_edge_attributes(k_subGraph, 'edge_name')[(edgeRightNode, edgeLeftNode)]
                        edgeList.append(edgeName)

                structPath = [item for sublist in zip(tempPath, edgeList) for item in sublist]
                structPath.append(tempPath[-1])

                structTable.append(torch.tensor(structPath).to(self.device))


        return structTable

    def forward(self, batch_h, batch_r, batch_t=None):
        # (h,r,t)
        r = self.unitized(self.R(batch_r))
        sequence = self.genInputSeqence(batch_h)
        sequence = torch.cat([sequence, r.view(-1,self.entity_dim)], dim=0)  # 在最后一个位置加入需要预测的关系
        x = self.outputLayer(sequence)[-1].view(-1,self.entity_dim)
        x = self.unitized(x) * self.c
        # x = F.relu(x)  # deletable
        entities_embedding = self.unitized(self.E.weight)
        x = torch.mm(x, entities_embedding.transpose(1, 0))  # *self.c  # c is important
        # x += self.b.expand_as(x)  # deletable
        y = torch.sigmoid(x)

        return self.loss(y, batch_t), y


class ConvELoss(BaseModel):
    def __init__(self, device, label_smoothing, entity_cnt):
        super().__init__()
        self.device = device
        self.loss = torch.nn.BCELoss(reduction='sum')
        self.label_smoothing = label_smoothing
        self.entity_cnt = entity_cnt

    def forward(self, batch_p, batch_t=None):
        batch_size = batch_p.shape[0]
        loss = None
        if batch_t is not None:
            batch_e = torch.zeros(batch_size, self.entity_cnt).to(self.device).scatter_(1, batch_t.view(-1, 1), 1)
            batch_e = (1.0 - self.label_smoothing) * batch_e + self.label_smoothing / self.entity_cnt
            loss = self.loss(batch_p, batch_e) / batch_size
            return loss

#
if __name__ == '__main__':
    # 构造有向图
    G = nx.DiGraph()

    # 添加节点
    G.add_node('计算机科学')
    G.add_node('数据结构')
    G.add_node('算法')
    G.add_node('机器学习')
    G.add_node('深度学习')
    G.add_node('神经网络')

    # 添加有向边，并为边添加属性
    G.add_edge('计算机科学', '数据结构')
    G.add_edge('数据结构', '算法', weight=2)
    G.add_edge('算法', '机器学习')
    G.add_edge('机器学习', '深度学习', weight=3)
    G.add_edge('深度学习', '神经网络')

    # 定义节点位置和标签
    pos = nx.spring_layout(G, seed=1)
    labels = {
        '计算机科学': '计算机科学',
        '数据结构': '数据结构',
        '算法': '算法',
        '机器学习': '机器学习',
        '深度学习': '深度学习',
        '神经网络': '神经网络'
    }

    # 绘制有向图
    nx.draw_networkx(G, pos, labels=labels, font_size=14, font_family='sans-serif')

    # 绘制边带权值标签
    edge_labels = nx.get_edge_attributes(G, 'weight')
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=12)

    # 显示图形
    plt.axis('off')
    plt.show()


    center_node = 'A'
    order = 1
    k_subgraph = nx.ego_graph(G, n=center_node, radius=order)

    nx.draw(k_subgraph, node_size = 30, with_labels=True)  # 绘制节点


    plt.show()  # 显示图像

    print(k_subgraph.nodes())