import torch
from sklearn.metrics import accuracy_score
import numpy as np
import time
from dgl.dataloading import GraphDataLoader
from tqdm import tqdm
from datetime import datetime
import csv
import os

from load_data import SEED_dependent_classify, SEED_independent_classify, DEAP_classify, DEAP_independent
from MTL_MGAWS.model.SupervisedTrainModel import SupervisedTrainModel
from params import build_args
from utils import set_random_seed, cacl_acc


def test(epoch, model, test_loader, sub, session, best_test_acc, best_epoch, save_model_dir, device):
    model.testmode = True
    model.eval()

    test_loss_list = []
    test_label_list = []
    test_prec_list = []
    for batch_g, label in test_loader:
        batch_g = batch_g.to(device)
        label = label.to(device)

        feat = batch_g.ndata["x"]
        a = feat.cpu().numpy()
        out, loss = model(batch_g, feat, label)

        prec = out.argmax(-1)

        test_loss_list.append(loss.item())
        test_prec_list.append(prec.cpu().numpy())
        test_label_list.append(label.cpu().numpy())

    y_true = np.concatenate(test_label_list, axis=0)
    y_pred = np.concatenate(test_prec_list, axis=0)
    test_acc = accuracy_score(y_true, y_pred)

    if test_acc > best_test_acc:
        best_test_acc = test_acc
        best_epoch = epoch
        ck = {}
        ck['epoch'] = epoch
        ck['model'] = model.state_dict()
        ck['acc'] = test_acc
        torch.save(ck, f'{save_model_dir}/checkpoint_s{str(sub+1).zfill(2)}_{session+1}.pkl')

    model.testmode = False
    return test_loss_list, test_acc, best_test_acc, best_epoch


def train(model, train_loader, test_loader, sub, k_fold, optimizer, scheduler, max_epoch, save_model_dir, device):
    best_test_acc, best_epoch = 0, 0
    test_loss_list, test_acc = None, None
    # train
    epoch_iter = tqdm(range(max_epoch))
    for epoch in epoch_iter:
        model.train()
        loss_list = []
        cm_loss_list = []
        fm_loss_list = []
        class_loss_list = []
        label_list = []
        prec_list = []
        model.train()

        for batch in train_loader:
            batch_g, label = batch
            batch_g = batch_g.to(device)
            label = label.to(device)
            feat = batch_g.ndata["x"]

            out, loss, [cm_loss, fm_loss, class_loss] = model(batch_g, feat, label)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            prec = out.argmax(-1)
            loss_list.append(loss.item())
            cm_loss_list.append(cm_loss.item())
            fm_loss_list.append(fm_loss.item())
            class_loss_list.append(class_loss.item())
            prec_list.append(prec.cpu().numpy())
            label_list.append(label.cpu().numpy())

        scheduler.step(np.mean(loss_list))

        y_true = np.concatenate(label_list, axis=0)
        y_pred = np.concatenate(prec_list, axis=0)
        acc = accuracy_score(y_true, y_pred)

        epoch_iter.set_description(f"Epoch {epoch} | train_loss: {np.mean(loss_list):.6f}| cm_loss: {np.mean(cm_loss_list):.6f}| "
                                   f"fm_loss: {np.mean(fm_loss_list):.6f}| class_loss: {np.mean(class_loss_list):.6f}| acc: {acc:.4f}")

        # test
        test_loss_list, test_acc, best_test_acc, best_epoch = test(epoch, model, test_loader, sub, k_fold, best_test_acc, best_epoch, save_model_dir, device)


    print(f"| test_loss: {np.mean(test_loss_list):.6f}| last_test_acc: {test_acc:.4f}|| best_test_acc: {best_test_acc:.4f}| best_epoch: {best_epoch}")

    return test_acc, best_test_acc

def main(args):
    dataset_name = args.dataset
    data_path = args.data_path
    batch_size = args.batch_size
    num_class = args.num_class
    max_epoch = args.max_epoch
    cheb_k = args.cheb_k
    mask_rate = args.mask_rate
    lr = args.lr
    weight_decay = args.weight_decay
    seed = args.seed
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

    save_model_dir = os.path.join(args.save_model_path, f'{dataset_name}_supervised_jointly_train')
    os.makedirs(save_model_dir, exist_ok=True)
    result_dir = os.path.join(args.result_path, f'{dataset_name}.txt')
    with open(result_dir, 'a+', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow([f'\n\n k = {args.cheb_k}  mask_rate={args.mask_rate}'])
        writer.writerow(['---------- ' + str(datetime.now()) + ' ----------'])

    if dataset_name == 'SEED':
        acc_list = []
        for sub in range(15):
            for session in range(3):
                print(f'Loading {dataset_name}-sub_{sub + 1}-day_{session + 1} dataset .....')
                start = time.time()
                train_dataset = SEED_dependent_classify(data_path, sub=sub, session=session, train=True)
                test_dataset = SEED_dependent_classify(data_path, sub=sub, session=session, train=False)
                train_loader = GraphDataLoader(train_dataset, batch_size=batch_size, shuffle=True)
                test_loader = GraphDataLoader(test_dataset, batch_size=batch_size, shuffle=False)
                print(f'Cost {np.around(time.time() - start, 4)} s')

                set_random_seed(seed)
                model = SupervisedTrainModel(input_dim=5, hidden_dim=32, num_class=num_class, cheb_k=cheb_k, mask_rate=mask_rate, testmode=False)
                model.to(device)
                optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
                scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=10,
                                                                       verbose=True, threshold=0.0001,
                                                                       threshold_mode='rel', cooldown=1, min_lr=0,
                                                                       eps=1e-8)
                last_test_acc, best_test_acc = train(model, train_loader, test_loader, sub, session, optimizer, scheduler, max_epoch, save_model_dir, device)
                acc_list.append(best_test_acc)

                with open(result_dir, 'a+', newline='') as csvfile:
                    writer = csv.writer(csvfile)
                    writer.writerow(['s%02d' % (sub + 1), ' day-{}'.format(session + 1), ' acc-%f' % best_test_acc])

            print(f"####### {dataset_name} ######Test_acc: {np.mean(acc_list):.4f}±{np.std(acc_list):.4f}")
            with open(result_dir, 'a+', newline='') as csvfile:
                writer = csv.writer(csvfile)
                writer.writerow([f"####### {dataset_name} ######Test_acc: {np.mean(acc_list):.4f}±{np.std(acc_list):.4f} "])

    elif dataset_name == 'SEED_indep':
        acc_list = []
        for sub in range(15):
            for session in range(3):
                set_random_seed(seed)
                print(f'Testing {dataset_name}-sub_{sub + 1}-day_{session + 1} dataset .....')
                start = time.time()
                train_dataset = SEED_independent_classify(data_path, sub=sub, session=session, train=True)
                test_dataset = SEED_independent_classify(data_path, sub=sub, session=session, train=False)
                print(f'Cost {np.around(time.time() - start, 4)} s')
                train_loader = GraphDataLoader(train_dataset, batch_size=batch_size, shuffle=True)
                test_loader = GraphDataLoader(test_dataset, batch_size=batch_size, shuffle=False)


                model = SupervisedTrainModel(input_dim=5, hidden_dim=32, num_class=num_class, cheb_k=cheb_k,
                                             mask_rate=mask_rate, testmode=False)
                model.to(device)
                optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
                scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=10,
                                                                       verbose=True, threshold=0.0001,
                                                                       threshold_mode='rel', cooldown=1, min_lr=0,
                                                                       eps=1e-8)
                last_test_acc, best_test_acc = train(model, train_loader, test_loader, sub, session, optimizer,
                                                     scheduler, max_epoch, save_model_dir, device)
                acc_list.append(best_test_acc)

                with open(result_dir, 'a+', newline='') as csvfile:
                    writer = csv.writer(csvfile)
                    writer.writerow(['s%02d' % (sub + 1), ' day-{}'.format(session + 1), ' acc-%f' % best_test_acc])

            print(f"####### {dataset_name} ######Test_acc: {np.mean(acc_list):.4f}±{np.std(acc_list):.4f}")
            with open(result_dir, 'a+', newline='') as csvfile:
                writer = csv.writer(csvfile)
                writer.writerow([f"####### {dataset_name} ######Test_acc: {np.mean(acc_list):.4f}±{np.std(acc_list):.4f} "])

    elif dataset_name == 'DEAP':
        acc_list = []
        for sub in range(32):
            sub_acc = []
            for k_fold in range(10):
                set_random_seed(seed)
                print(f'Testing {dataset_name}-sub_{sub + 1}-fold_{k_fold + 1} dataset .....')
                start = time.time()
                train_dataset = DEAP_classify(data_path, sub=sub, k_fold=k_fold, train=True, label_flag='arousal')
                test_dataset = DEAP_classify(data_path, sub=sub, k_fold=k_fold, train=False, label_flag='arousal')
                print(f'Cost {np.around(time.time() - start, 4)} s')
                train_loader = GraphDataLoader(train_dataset, batch_size=batch_size, shuffle=True)
                test_loader = GraphDataLoader(test_dataset, batch_size=batch_size, shuffle=False)


                model = SupervisedTrainModel(input_dim=4, hidden_dim=32, num_class=num_class, cheb_k=cheb_k, mask_rate=mask_rate, testmode=False)
                model.to(device)
                optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
                scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=10,
                                                                       verbose=True, threshold=0.0001,
                                                                       threshold_mode='rel', cooldown=1, min_lr=0,
                                                                       eps=1e-8)
                last_test_acc, best_test_acc = train(model, train_loader, test_loader, sub, k_fold, optimizer, scheduler, max_epoch, save_model_dir, device)
                sub_acc.append(best_test_acc)

                with open(result_dir, 'a+', newline='') as csvfile:
                    writer = csv.writer(csvfile)
                    writer.writerow(['s%02d' % (sub + 1), ' fold-{}'.format(k_fold + 1), ' acc-%f' % best_test_acc])

            acc_list.append(np.mean(sub_acc))
            print(f"####### {dataset_name} ######Test_acc: {np.mean(acc_list):.4f}±{np.std(acc_list):.4f}")
            with open(result_dir, 'a+', newline='') as csvfile:
                writer = csv.writer(csvfile)
                writer.writerow([f"####### {dataset_name} ######Sub_acc: {np.mean(sub_acc):.4f} "])
                writer.writerow([f"####### {dataset_name} ######Test_acc: {np.mean(acc_list):.4f}±{np.std(acc_list):.4f} "])

    elif dataset_name == 'DEAP_indep':
        acc_list = []
        for sub in range(32):
            sub_acc = []
            for k_fold in range(1):
                set_random_seed(seed)
                print(f'Testing {dataset_name}-sub_{sub + 1}-fold_{k_fold + 1} dataset .....')
                start = time.time()
                train_dataset = DEAP_independent(data_path, sub=sub, train=True, label_flag='arousal')
                test_dataset = DEAP_independent(data_path, sub=sub, train=False, label_flag='arousal')
                print(f'Cost {np.around(time.time() - start, 4)} s')
                train_loader = GraphDataLoader(train_dataset, batch_size=batch_size, shuffle=True)
                test_loader = GraphDataLoader(test_dataset, batch_size=batch_size, shuffle=False)


                model = SupervisedTrainModel(input_dim=4, hidden_dim=32, num_class=num_class, cheb_k=cheb_k, mask_rate=mask_rate, testmode=False)
                model.to(device)
                optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
                scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=10,
                                                                       verbose=True, threshold=0.0001,
                                                                       threshold_mode='rel', cooldown=1, min_lr=0,
                                                                       eps=1e-8)
                last_test_acc, best_test_acc = train(model, train_loader, test_loader, sub, k_fold, optimizer, scheduler, max_epoch, save_model_dir, device)
                sub_acc.append(best_test_acc)

                with open(result_dir, 'a+', newline='') as csvfile:
                    writer = csv.writer(csvfile)
                    writer.writerow(['s%02d' % (sub + 1), ' fold-{}'.format(k_fold + 1), ' acc-%f' % best_test_acc])

            acc_list.append(np.mean(sub_acc))
            print(f"####### {dataset_name} ######Test_acc: {np.mean(acc_list):.4f}±{np.std(acc_list):.4f}")
            with open(result_dir, 'a+', newline='') as csvfile:
                writer = csv.writer(csvfile)
                # writer.writerow([f"####### {dataset_name} ######Sub_acc: {np.mean(sub_acc):.4f} "])
                writer.writerow([f"####### {dataset_name} ######Test_acc: {np.mean(acc_list):.4f}±{np.std(acc_list):.4f} "])





if __name__ == "__main__":
    args = build_args()

    print(args)
    main(args)
    # cacl_acc(args)

























