import argparse

N_FEATURES_ALL = 320
N_FEATURES_GLOVE = 300
NUM_FOLDS = 5
HATE_ALL_FILE = "./data/users_hate_all.content"
HATE_GLOVE_FILE = "./data/users_hate_glove.content"
SUSPENSION_ALL_FILE = "./data/users_suspended_all.content"
SUSPENSION_GLOVE_FILE = "./data/users_suspended_glove.content"
EDGES_FILE = "./data/users.edges"
NUM_NODES = 100386
ACTIVITY_CSV = "./data/users_activity.csv"
MODE = "hate"
NETWORK = "SAGE"
DROPOUT = 0.1
ATTN_DROPOUT = 0
ATTN_HEADS = 2
NUM_LAYERS = 2
EPOCHS = 2
HIDDEN_FEATURES = 256
IN_FEATURES = N_FEATURES_ALL
FEATURES_FILE = HATE_ALL_FILE

def run_parser():

    global EPOCHS,HIDDEN_FEATURES,IN_FEATURES,FEATURES_FILE, MODE,NETWORK,DROPOUT,ATTN_DROPOUT,NUM_LAYERS,ATTN_HEADS

    parser = argparse.ArgumentParser(description="add your runtime options")

    parser.add_argument("--mode" , type = str , default = 'hate' , action= 'store' , help = "hate/suspend")
    parser.add_argument("--features" , type = str , default='all' , action='store' , help = "glove/all")
    parser.add_argument("--network" , type=str , default='SAGE' , action='store' , help='NAIVE/PROP/GAT/SAGE/GIN')
    parser.add_argument("--hidden" , type = int , default=256 , action='store' , help = "hidden dimensionality")
    parser.add_argument("--layers",type = int , default=2 , action='store' , help="how many layers (recommended 2)")
    parser.add_argument("--batchsize" , type = int , default=256 , action='store' , help='batch size for stochastic training')
    parser.add_argument("--epochs" , type = int , default=10 , action='store' , help = "number of epochs")
    parser.add_argument("--dropout" , type = float , default=0.1 , action='store' , help= "dropout")
    parser.add_argument("--attndropout" , type = float , default=0 , action='store' , help="attention dropout -- GAT")
    parser.add_argument("--attnheads" , type = int , default=2 , action='store' , help = "attention heads -- GAT")
    args = parser.parse_args()
    EPOCHS = args.epochs
    HIDDEN_FEATURES = args.hidden

    assert args.mode in ['hate' , 'suspend'] , "--mode = hate/suspend"
    assert args.features in ['glove' , 'all'] , "--features = glove/all"
    assert args.network in ['NAIVE' , 'PROP' , 'GAT' , 'SAGE' , 'GIN'] , "--network NAIVE/PROP/GAT/SAGE/GIN"


    if args.features == 'all':
        IN_FEATURES = N_FEATURES_ALL
        if args.mode == 'hate':
            FEATURES_FILE = HATE_ALL_FILE
        else:
            FEATURES_FILE = SUSPENSION_ALL_FILE
    elif args.features == 'glove':
        IN_FEATURES = N_FEATURES_GLOVE
        if args.mode == 'hate':
            FEATURES_FILE = HATE_GLOVE_FILE
        else:
            FEATURES_FILE = SUSPENSION_GLOVE_FILE

    MODE = args.mode
    NETWORK = args.network
    DROPOUT = args.dropout
    ATTN_DROPOUT = args.attndropout
    NUM_LAYERS = args.layers
    ATTN_HEADS = args.attnheads





