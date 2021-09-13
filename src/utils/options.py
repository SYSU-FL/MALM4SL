import argparse
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=200,
                        help="number of rounds of training")
    parser.add_argument('--num_users', type=int, default=50,
                        help="number of users: K")
    parser.add_argument('--frac', type=float, default=1, 
                        help='the fraction of clients: C')
    parser.add_argument('--local_ep', type=int, default=1,
                        help="the number of local epochs: E")
    parser.add_argument('--local_bs', type=int, default=128,
                        help="local batch size: B")
    parser.add_argument('--lr', type=float, default=0.1,
                        help='learning rate')
    parser.add_argument('--momentum', type=float, default=0.9,
                        help='SGD momentum (default: 0.5)')
    parser.add_argument('--balanced', action='store_true', help="whether to use the balanced data")
    
    parser.add_argument('--mix_num', type=int, default=3, help="the number of labels to mix up")

    parser.add_argument('--dp', action='store_true', help="whether to use the differential privacy")
    parser.add_argument('--dcor', action='store_true', help="whether to use the distance correlation")

    parser.add_argument('--gpu', type=int, default=0, help="the index of gpus")
    parser.add_argument('--seed', type=int, default=0, help='random seed')#42
    parser.add_argument('--verbose', type=int, default=1, help='verbose')
    parser.add_argument('--name', type=str)
    parser.add_argument('--cifar100', action='store_true', help="whether to use cifar100")
    parser.add_argument('--mnist', action='store_true', help="whether to use mnist")
    parser.add_argument('--model', type=str, default='cnnmnist', help="the model used to train")
    parser.add_argument('--google_split', action='store_true', help="the spliting way")

    parser.add_argument('--local_rank', type=int)
    parser.add_argument('--ip', type=str, default="", help="socket server ip")
    parser.add_argument('--port', type=int, default=10090, help="socket server port")

    parser.add_argument('--resume', action='store_true', help="whether to load pre train model")
    parser.add_argument('--reload_path', type=str, default="", help="the path to reload models")
    
    args = parser.parse_args()
    return args

