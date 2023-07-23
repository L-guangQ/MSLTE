import argparse

def build_args():
    parser = argparse.ArgumentParser(description="")
    parser.add_argument("--seed", type=int, nargs="+", default=2023)
    parser.add_argument("--dataset", type=str, default="SEED_indep", choices=['SEED', 'SEED_indep', 'DEAP', 'DEAP_indep'])
    parser.add_argument("--data_path", type=str, default=r'D:\EEGdata\seed\feature_each_trial')
    parser.add_argument("--num_class", type=int, default=3)
    parser.add_argument("--save_model_path", type=str, default='./save_model')
    parser.add_argument("--result_path", type=str, default=r'./result')
    parser.add_argument("--max_epoch", type=int, default=200, help="number of training epochs")
    parser.add_argument("--use_scheduler", action="store_true", default=False)

    parser.add_argument("--batch_size", type=int, default=128, choices=[128, 1024])
    parser.add_argument("--lr", type=float, default=0.005, help="learning rate", choices=[0.05, 0.01, 0.005, 0.0005])
    parser.add_argument("--weight_decay", type=float, default=8e-5, help="weight decay")
    parser.add_argument("--cheb_k", type=int, default=2, help='Chebyshev polynomial order')
    parser.add_argument("--mask_rate", type=float, default=0.7, help="feature mask rate", choices=['0.2-0.8'])


    args = parser.parse_args()
    return args