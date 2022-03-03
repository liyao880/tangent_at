import os
os.chdir(r'D:\yaoli\tangent')
import math
import torch
import argparse
import numpy as np
import torch.nn as nn
from tqdm import tqdm
from tangent import check_tangent
from train_ae import Autoencoder
from setup.utils import loaddata, loadmodel
from setup.setup_pgd import to_var, adv_train, pred_batch, LinfPGDAttack

# Empirical study of tangent components range
def main(args):
    result_dir = os.path.join(args['root'], 'tangent', 'results')
    os.makedirs(result_dir, exist_ok=True)
    
    print('==> Loading data..')
    train_loader, _ = loaddata(args)
    
    print('==> Loading model..')
    model = loadmodel(args)
    autoencoder = load_autoencoder(args)
    
    print('==> Generating components..')
    compute_tangent_components(args, model, autoencoder, train_loader, result_dir)
    
    return


def load_autoencoder(args):
    autoencoder = Autoencoder(args['dim'])
    if args['ae_load']:
        print("Loading pre-trained models {}".format(args['ae_load']))
        state_dict = torch.load(os.path.join(args['model_folder'],args['ae_load']))
        autoencoder.load_state_dict(state_dict)
    if torch.cuda.is_available():
        autoencoder = autoencoder.cuda()
        print("Model moved to GPU in order to speed up training.")
    return autoencoder


def compute_tangent_components(args, model, autoencoder, train_loader, result_dir):
    model = model.cuda()
        
    adversary = LinfPGDAttack(epsilon=args['epsilon'], k=args['num_k'], a=args['alpha'])
    criterion = nn.CrossEntropyLoss()
    
    X_ori = []
    X_advs = []
    X_components = []
    labels = []
    for x, target in tqdm(train_loader):            
        x, target = to_var(x), to_var(target)
        target_pred = pred_batch(x, model)
        x_adv = adv_train(x, target_pred, model, criterion, adversary)
        components = check_tangent(autoencoder, x, x_adv, k=10)
        X_advs.append(x_adv.detach().cpu().numpy())
        X_components.append(components)
        labels.append(target.detach().cpu().numpy())
        X_ori.append(x.detach().cpu().numpy())
    X_advs = np.concatenate(X_advs, axis=0)
    X_components = np.concatenate(X_components)
    labels = np.concatenate(labels)
    X_ori = np.concatenate(X_ori, axis=0)
    print("Components Stat:")
    print("Max: {:.3f}".format(X_components.max()))
    print("Min: {:.3f}".format(X_components.min()))
    print("Mean: {:.3f}".format(X_components.mean()))
    
    np.save(os.path.join(result_dir, 'X_adv_'+args['dataset']+'.npy'), X_advs)
    np.save(os.path.join(result_dir, 'comp_'+args['dataset']+'.npy'), X_components)
    np.save(os.path.join(result_dir, 'labels_'+args['dataset']+'.npy'), labels)
    np.save(os.path.join(result_dir, 'X_ori_'+args['dataset']+'.npy'), X_ori)
    
    return


def testClassifier(X_adv, labels, model, batch_size=100):
    model = model.cuda()
    model.eval()
    correct_cnt = 0
    total_cnt = 0
    n = X_adv.shape[0]
    for i in range(math.floor(n/batch_size)):
        batch = X_adv[i*batch_size:(i+1)*batch_size]
        target = labels[i*batch_size:(i+1)*batch_size]
        with torch.no_grad(): 
            batch = torch.tensor(batch).cuda()
            target = torch.tensor(target).cuda()
            out = model(batch)
            _, pred_label = torch.max(out.data, 1)
            total_cnt += batch.data.size()[0]
            correct_cnt += (pred_label == target.data).sum()
    acc = float(correct_cnt.double()/total_cnt)
    print("The prediction accuracy on data is {}".format(acc))
    return acc


def explore_adv_comp(model, result_dir):
    X_ori = np.load(os.path.join(result_dir, 'X_ori_'+args['dataset']+'.npy'))
    X_adv = np.load(os.path.join(result_dir, 'X_adv_'+args['dataset']+'.npy'))
    labels = np.load(os.path.join(result_dir, 'labels_'+args['dataset']+'.npy'))
    components = np.load(os.path.join(result_dir, 'comp_'+args['dataset']+'.npy'))
    
    # Rank the images based on components values
    ind = np.argsort(components)
    components = components[ind]
    X_ori = X_ori[ind]
    X_adv = X_adv[ind]
    labels = labels[ind]
    
    testClassifier(X_adv, labels, model)
    # Get the change of clean accuracy and robust accuracy after adversarial training for each batch
    
    
    
    return

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Training defense models')
    parser.add_argument("-d", '--dataset', choices=["mnist", "cifar10","stl10","tiny"], default="cifar10")   
    parser.add_argument("--init", default='cifar10_clean')
    parser.add_argument("--root", default=r'D:\yaoli')
    parser.add_argument("--model_folder", default='./models',
                        help="Path to the folder that contains checkpoint.")
    parser.add_argument("--ae_load", default='ae_loss0.589.pt',
                        help="name of the autoencoder to load.")
    parser.add_argument("--train_shuffle", action="store_true",  default=False,
                        help="shuffle in training or not")    
    args = vars(parser.parse_args())
    if args['dataset'] == 'mnist':
        args['alpha'] = 0.02
        args['num_k'] = 40
        args['epsilon'] = 0.3
        args['batch_size'] = 100
    elif args['dataset'] == 'cifar10':
        args['alpha'] = 0.01
        args['num_k'] = 20
        args['epsilon'] = 0.03
        args['batch_size'] = 100
        args['dim'] = 128
    elif args['dataset'] == 'stl10':
        args['alpha'] = 0.0156
        args['num_k'] = 20
        args['epsilon'] = 0.03
        args['batch_size'] = 64
    elif args['dataset'] == 'tiny':
        args['alpha'] = 0.002
        args['num_k'] = 10
        args['epsilon'] = 0.01
        args['batch_size'] = 128
        args['num_gpu'] = 2
    else:
        print('invalid dataset')
    print(args)
    main(args)