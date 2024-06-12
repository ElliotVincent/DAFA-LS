from src.utils.paths import RESULTS_PATH, CONFIGS_PATH
import numpy as np
import torch
import argparse
import os
import yaml
import tqdm
import json
from src.utils.utils import Logger, coerce_to_path_and_create_dir, coerce_to_path_and_check_exist, get_model, get_loader, get_criterion
from src.utils.metrics import BinaryClassifEval


def iterate(batch, device, model, criterion, metrics, single=False, pixel_wise=False, semantic=False,
            mode='train', month_id=None, year_id=None):
    if pixel_wise:
        data, label = batch
        label = label.squeeze(1).to(device)
        curr_labels = label
        positions = torch.tensor(range(96), device=device)[None].expand(data.shape[0], -1)
        if mode == 'eval':
            data = data.squeeze(0)
            logits = None
            for k in range(data.shape[0] // 128 + int(data.shape[0] % 128 != 0)):
                curr_data = data[k * 128: (k + 1) * 128]
                curr_pos = positions[k * 128: (k + 1) * 128]
                curr_logits = model(curr_data, batch_positions=curr_pos)
                if logits is None:
                    logits = curr_logits
                else:
                    logits = torch.cat([logits, curr_logits], dim=0)
            curr_preds = torch.argmax(logits, dim=1)
            preds = (curr_preds.sum() / curr_preds.shape[0] > 0.5).int()[None]
            logits = logits[curr_preds == preds].mean(0, keepdim=True)
        else:
            logits = model(data, batch_positions=positions)
            preds = torch.argmax(logits, dim=1)
    elif single:
        data, positions, mask, label = batch
        label = label.squeeze(1).to(device)
        curr_labels = label
        if mode == 'eval':
            # B x T x C x H x W
            if month_id is None:
                data = data[:, -12:]
            else:
                data = data[:, month_id].unsqueeze(1)
            data = data.reshape(-1, data.shape[2], data.shape[3], data.shape[4])
            logits = model(data, mask=mask)
            logits = logits.reshape(positions.shape[0], -1, 2)
            preds = torch.argmax(logits, dim=2)
            preds = torch.mode(preds, dim=1)[0]
            logits = logits.mean(1)
        else:
            # B x C x H x W
            logits = model(data, mask=mask)
            preds = torch.argmax(logits, dim=1)
    else:
        data, positions, mask, label = batch
        label = label.squeeze(1).to(device)
        curr_labels = label
        if year_id is not None:
            data = data[:, torch.div(positions.squeeze(0), 12, rounding_mode='floor') == year_id]
            positions = positions[:, torch.div(positions.squeeze(0), 12, rounding_mode='floor') == year_id]
        logits = model(data, batch_positions=positions, mask=mask.int())
        if semantic:
            preds = torch.argmax(logits[:, 1:], dim=1).flatten(1) + 1
            mask = mask.flatten(1).to(device)
            preds = [preds[batch_id][mask[batch_id] == 1] for batch_id in range(preds.shape[0])]
            try:
                preds = [torch.mode(curr_preds)[0] for curr_preds in preds]
            except:
                print("Only zeros in mask!")
                preds = [torch.tensor([1], device=device) for _ in range(len(preds))]
            preds = torch.stack(preds, dim=0) - 1
            label = ((label == 2).sum((1, 2)) > 0).int()
            if mode == 'eval':
                logits = [logits[batch_id][1:].flatten(1)[:, mask[batch_id] == 1] for batch_id in range(preds.shape[0])]
                logits = torch.stack([curr_logits[:, torch.argmax(curr_logits, dim=0) == preds].mean(1) for curr_logits in logits])
                curr_labels = label.long()
        else:
            preds = torch.argmax(logits, dim=1)
    if criterion._get_name() == 'BCEWithLogitsLoss':
        curr_labels = torch.nn.functional.one_hot(label, num_classes=2).float()
    loss = criterion(logits, curr_labels)
    metrics.add(preds, label, loss, logits)
    return loss


def main(config, res_dir, tag, fold):
    print(json.dumps(config, indent=2))
    if torch.cuda.is_available():
        type_device = "cuda"
        nb_device = torch.cuda.device_count()
    else:
        type_device = "cpu"
        nb_device = None
    print("Using {} device, nb_device is {}".format(type_device, nb_device))
    device = torch.device('cuda')

    n_epochs = config['training']['n_epochs']
    num_gpus = config['training'].get('num_gpus', 1)
    lr = config['training']['optimizer']['lr']
    weight_decay = config['training']['optimizer']['weight_decay']
    gamma = config['training']['scheduler']['gamma']
    warmup = config['training']['scheduler']['warmup']
    num_iter_per_step = config['training']['scheduler']['num_iter_per_step']
    min_lr = config['training']['scheduler']['min_lr']

    train_loader = get_loader(config, 'train', fold)
    val_loader = get_loader(config, 'val', fold, batch_size=1, single=False, augment=False)

    model = get_model(config)
    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model has {n_parameters} parameters.")
    config['model']['n_parameters'] = n_parameters
    with open(os.path.join(res_dir, "conf.json"), "w") as file:
        file.write(json.dumps(config, indent=4))
    model = torch.nn.DataParallel(model.to(device), device_ids=[k for k in range(num_gpus)])
    optimizer = torch.optim.AdamW(model.parameters(), lr=0., weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma)
    criterion = get_criterion(config, device)
    train_metrics = BinaryClassifEval(device)
    train_logger = Logger('Train', 'logs')
    val_logger = Logger('Val', 'logs')
    viz = None
    n_iter = 0
    curr_step = 0
    best_m_acc = 0
    model.train()
    n_epochs = min(n_epochs, num_iter_per_step[-1] // (train_loader.dataset.len // train_loader.batch_size) + 1)
    for epoch in range(1, n_epochs + 1):
        print(f"Epoch {epoch}/{n_epochs}.")
        for i, batch in tqdm.tqdm(enumerate(train_loader), total=train_loader.dataset.len // train_loader.batch_size):
            optimizer.zero_grad()
            loss = iterate(batch, device, model, criterion, train_metrics, single=config['dataset'].get('single', False),
                           pixel_wise=config['dataset'].get('pixel_wise', False),
                           semantic=config['dataset'].get('semantic', False))
            loss.backward()
            optimizer.step()
            n_iter += 1

            if n_iter <= warmup:
                rate = lr * min(1., n_iter / warmup)
                for p in optimizer.param_groups:
                    p['lr'] = rate
                scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma)

            if curr_step < len(num_iter_per_step) and n_iter > warmup and n_iter % num_iter_per_step[curr_step] == 0:
                scheduler.step()
                curr_step += 1

            torch.save(
                {
                    "iter": n_iter,
                    "state_dict": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                },
                os.path.join(
                    res_dir, "model_last.pth.tar"
                ))

            if config['dataset'].get('pixel_wise', False):
                if n_iter % 500 == 0:
                    print(f'Train iter: {n_iter}')
                    metrics = train_metrics.compute_metrics()
                    train_metrics = BinaryClassifEval(device)
                    train_logger.update(viz, n_iter, res_dir, metrics)
                    print()

                    print(f'Val iter: {n_iter}')
                    model.eval()
                    val_metrics = BinaryClassifEval(device)
                    with torch.no_grad():
                        for j, batch in tqdm.tqdm(enumerate(val_loader),
                                                  total=val_loader.dataset.len // val_loader.batch_size):
                            iterate(batch, device, model, criterion, val_metrics,
                                    single=config['dataset'].get('single', False),
                                    pixel_wise=config['dataset'].get('pixel_wise', False), mode='eval',
                                    semantic=config['dataset'].get('semantic', False))
                    metrics = val_metrics.compute_metrics()
                    metrics['lr'] = scheduler.get_last_lr()[0]
                    val_logger.update(viz, n_iter, res_dir, metrics)
                    m_acc = metrics['mean_acc']
                    model.train()
                    print()

                    if best_m_acc <= m_acc:
                        best_m_acc = m_acc
                        torch.save(
                            {
                                "iter": n_iter,
                                "state_dict": model.state_dict(),
                                "optimizer": optimizer.state_dict(),
                            },
                            os.path.join(
                                res_dir, "model_macc.pth.tar"
                            ))

        if not config['dataset'].get('pixel_wise', False):

            print(f'Train iter: {n_iter}')
            metrics = train_metrics.compute_metrics()
            train_metrics = BinaryClassifEval(device)
            train_logger.update(viz, n_iter, res_dir, metrics)
            print()

            print(f'Val iter: {n_iter}')
            model.eval()
            val_metrics = BinaryClassifEval(device)
            with torch.no_grad():
                for j, batch in tqdm.tqdm(enumerate(val_loader), total=val_loader.dataset.len // val_loader.batch_size):
                    iterate(batch, device, model, criterion, val_metrics, single=config['dataset'].get('single', False),
                            pixel_wise=config['dataset'].get('pixel_wise', False), mode='eval',
                            semantic=config['dataset'].get('semantic', False))
            metrics = val_metrics.compute_metrics()
            metrics['lr'] = scheduler.get_last_lr()[0]
            val_logger.update(viz, n_iter, res_dir, metrics)
            m_acc = metrics['mean_acc']
            model.train()
            print()

            if best_m_acc <= m_acc:
                best_m_acc = m_acc
                torch.save(
                    {
                        "iter": n_iter,
                        "state_dict": model.state_dict(),
                        "optimizer": optimizer.state_dict(),
                    },
                    os.path.join(
                        res_dir, "model_macc.pth.tar"
                    ))

        if n_iter > warmup and (optimizer.param_groups[0]['lr'] < min_lr or curr_step == len(num_iter_per_step)):
            break


def eval(res_dir, ckpt='macc', split='test', month_id=None):
    config = json.load(open(os.path.join(res_dir, "conf.json")))
    print(json.dumps(config, indent=2))
    if torch.cuda.is_available():
        type_device = "cuda"
        nb_device = torch.cuda.device_count()
    else:
        type_device = "cpu"
        nb_device = None
    print("Using {} device, nb_device is {}".format(type_device, nb_device))
    device = torch.device('cuda')
    loader = get_loader(config, split, None, augment=False, batch_size=1, single=False)
    model = get_model(config)
    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model has {n_parameters} parameters.")
    num_gpus = 1
    model = torch.nn.DataParallel(model.to(device), device_ids=[k for k in range(num_gpus)])
    model.load_state_dict(torch.load(os.path.join(res_dir, f'model_{ckpt}.pth.tar'))['state_dict'])
    model.eval()
    metrics = BinaryClassifEval(device)
    criterion = get_criterion(config, device)
    with torch.no_grad():
        for j, batch in tqdm.tqdm(enumerate(loader), total=loader.dataset.len//loader.batch_size):
            iterate(batch, device, model, criterion, metrics, single=config['dataset'].get('single', False),
                    pixel_wise=config['dataset'].get('pixel_wise', False), mode='eval',
                    semantic=config['dataset'].get('semantic', False),
                    month_id=month_id)
    metrics = metrics.compute_metrics()
    file_name = f'{split}_metrics_{ckpt}'
    file_name += f'y{month_id}' if month_id is not None else ''
    file_name += '_all.json'
    with open(os.path.join(res_dir, file_name), 'w') as file:
        file.write(json.dumps(metrics, indent=4))


def print_result(tag, ckpt='macc', split='test', month_id=None):
    path = os.path.join(RESULTS_PATH, tag)
    oa, f1, auroc, fpr = [], [], [], []
    num_runs = 5
    for run_id in range(1, num_runs + 1):
        file_name = f'{split}_metrics_{ckpt}'
        file_name += f'y{month_id}' if month_id is not None else ''
        file_name += '_all.json'
        curr_path = os.path.join(path, f'Fold_{run_id}', file_name)
        oa.append(json.load(open(curr_path))['acc'])
        f1.append(json.load(open(curr_path))['f1'])
        auroc.append(json.load(open(curr_path))['auroc'])
        fpr.append(json.load(open(curr_path))['fpr'])
    print(f'{tag} on {split} with {ckpt} ckpt:')
    print(f'   OA: {np.mean(oa):.2f} ({np.std(oa):.2f})')
    print(f'   F1: {np.mean(f1):.2f} ({np.std(f1):.2f})')
    print(f'   AUROC: {np.mean(auroc):.2f} ({np.std(auroc):.2f})')
    print(f'   FPR: {np.mean(fpr):.2f} ({np.std(fpr):.2f})')


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Pipeline to train a NN model specified by a YML config")
    parser.add_argument("-t", "--tag", nargs="?", type=str, required=True, help="Run tag of the experiment")
    parser.add_argument("-c", "--config", nargs="?", type=str, required=True, help="Config file name")
    args = parser.parse_args()

    print(f'Experiment tag is {args.tag}.')
    print(f'Configuration file is {args.config}.')

    assert args.tag is not None and args.config is not None
    config = coerce_to_path_and_check_exist(CONFIGS_PATH / args.config)
    with open(config) as fp:
        cfg = yaml.load(fp, Loader=yaml.FullLoader)

    tag = f'{args.tag}'
    num_runs = 5
    for fold in range(1, num_runs + 1):
        frozen = cfg['model'].get('frozen', False)
        coerce_to_path_and_create_dir(RESULTS_PATH / tag)
        run_id = f'Fold_{fold}'
        res_dir = RESULTS_PATH / tag / run_id
        coerce_to_path_and_create_dir(res_dir)
        main(cfg, res_dir, tag, fold)
        eval(res_dir, ckpt='macc', split='test')
    print_result(tag, ckpt='macc', split='test')
