import dgl
import config
import dataloading
import models
import numpy as np
import torch
from sklearn.model_selection import StratifiedKFold
from utils import SummaryRecorder


# loading the data
data, labels, edges = dataloading.load_features(config.FEATURES_FILE , config.EDGES_FILE , config.IN_FEATURES)
labels = [ 1 if x == 2 else 0 for x in labels ]
labels = np.array(labels)

def run_graph_training():
    # initializing the graph
    g = dgl.graph(data = edges , num_nodes=config.NUM_NODES)
    g.ndata['features']  = torch.from_numpy(data.astype(np.float32))
    g.ndata['labels'] = torch.from_numpy(np.array(labels).reshape(-1 , 1))
    all_nodes = dataloading.get_annotated_users(config.MODE)
    sampler = dgl.dataloading.MultiLayerFullNeighborSampler(config.NUM_LAYERS)

    # splitting the data
    splits = StratifiedKFold(config.NUM_FOLDS , shuffle=True).split(all_nodes , labels[all_nodes])

    # recorder init
    history = SummaryRecorder(config.NUM_FOLDS , config.EPOCHS)

    for fold_id , (train_index , test_index) in enumerate(splits):

        #initializing a new model
        model = models.StochasticNetwork()
        model = model.cuda()
        opt = torch.optim.Adam(model.parameters())

        train_nodes = all_nodes[train_index]
        train_labels = labels[train_index]

        validation_nodes = all_nodes[test_index]
        validation_labels = labels[test_index]

        loss_fn = torch.nn.CrossEntropyLoss()


        for epoch_id in range(config.EPOCHS):

            model.train()

            # load nodes in batches
            dataloader = dgl.dataloading.pytorch.NodeDataLoader(g, train_nodes, sampler, batch_size=256)
            total_batches = len(dataloader)


            for input_nodes, output_nodes, blocks in dataloader:

                # fetch the batch nodes and dependencies
                blocks = [b.to(torch.device('cuda')) for b in blocks]
                input_features = blocks[0].srcdata['features']
                true_labels = torch.flatten(blocks[-1].dstdata['labels'])

                # forward pass
                output_predictions = model(blocks, input_features)
                pred_labels = output_predictions

                # calculate loss
                loss = loss_fn(output_predictions , true_labels)

                # backward pass
                opt.zero_grad()
                loss.backward()
                opt.step()

                # metrics
                true_labels = true_labels.cpu().numpy()
                pred_labels = pred_labels.detach().cpu().numpy().argmax(axis = 1).flatten()

                history.add_evaluation("train",fold_id,epoch_id,loss.item(),y_true=true_labels,y_pred=pred_labels)

            """
            ############################################################
                                    EVALUATION
            ############################################################
            """
            model.eval()

            dataloader = dgl.dataloading.pytorch.NodeDataLoader(g, validation_nodes, sampler, batch_size=256)
            total_batches = len(dataloader)

            for input_nodes, output_nodes, blocks in dataloader:
                # fetch the batch nodes and dependencies
                blocks = [b.to(torch.device('cuda')) for b in blocks]
                input_features = blocks[0].srcdata['features']
                true_labels = torch.flatten(blocks[-1].dstdata['labels'])

                # forward pass
                output_predictions = model(blocks, input_features)
                pred_labels = output_predictions.detach().cpu().numpy().argmax(axis=1).flatten()

                loss = loss_fn(output_predictions , true_labels)
                true_labels = true_labels.cpu().numpy()

                history.add_evaluation("eval", fold_id, epoch_id, loss.item(), y_true=true_labels, y_pred=pred_labels)

    print(history.generate_summary())

if __name__ == '__main__':
    config.run_parser()
    if config.NETWORK in ['SAGE' , 'GAT' , 'GIN']:
        run_graph_training()