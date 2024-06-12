import torch
from sklearn.metrics import roc_curve, auc


class BinaryClassifEval:
    def __init__(self, device):
        super(BinaryClassifEval, self).__init__()
        self.n_classes = 2
        self.conf_matrix = torch.zeros((2, 2), device=device, dtype=torch.long)
        self.loss = 0
        self.count = 0
        self.labs = None
        self.preds = None
        self.logits = None

    def add(self, preds, labs, loss, logits):
        """
        :param preds: predictions, has shape B
        :param labs: shape B
        :return:
        """
        self.loss += loss.item()
        self.count += 1
        self.conf_matrix.index_put_((labs.flatten().long(),
                                     preds.flatten().long()),
                                    torch.ones(preds.flatten().size(0), device=preds.device, dtype=torch.long),
                                    accumulate=True)
        if self.labs == None:
            self.labs = labs.flatten().long()
            self.preds = preds.flatten().long()
            self.logits = logits.float()
        else:
            self.labs = torch.cat([self.labs, labs.flatten().long()])
            self.preds = torch.cat([self.preds, preds.flatten().long()])
            self.logits = torch.cat([self.logits, logits.float()])

    def compute_metrics(self):
        probabilities = torch.softmax(self.logits, dim=1)[:, 1]
        fpr, tpr, thresholds = roc_curve(self.labs.detach().cpu().numpy(),
                                                         probabilities.detach().cpu().numpy(),
                                                         pos_label=1)

        auroc = auc(fpr, tpr) * 100
        loss = self.loss / self.count
        conf_matrix = self.conf_matrix
        TP, TN, FP, FN = conf_matrix[1, 1], conf_matrix[0, 0], conf_matrix[0, 1], conf_matrix[1, 0]
        precision = TP / (TP+FP) * 100
        recall = TP / (TP+FN) * 100
        f1 = 2 * precision * recall / (precision + recall + 0.0000000001)
        acc = (torch.diag(conf_matrix).sum() / conf_matrix.sum()).item() * 100
        class_acc = [(conf_matrix[class_id, class_id] / max(conf_matrix[class_id, :].sum().item(), 1)).item() * 100
                     for class_id in range(self.n_classes)]
        m_acc = torch.tensor(class_acc).mean().item()
        fpr = (conf_matrix[0, 1] / (conf_matrix[0, 1] + conf_matrix[0, 0]) * 100).item()
        print('Loss: {:.3f}, Acc: {:.2f}, mAcc: {:.2f}'.format(loss, acc, m_acc))
        print('Per class ACC:')
        for class_id, class_name in enumerate(['Not looted', 'Looted']):
            print('      {}: {:.2f}'.format(class_name, class_acc[class_id]))
        return {'loss': loss, 'acc': acc, 'mean_acc': m_acc,
                'class_acc': class_acc, 'conf_mat': conf_matrix.tolist(), 'fpr': fpr,
                'auroc': auroc, 'f1': f1.item(), 'precision': precision.item(), 'recall': recall.item()}
