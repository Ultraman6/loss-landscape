import numpy as np
import torch
from torch import nn


def save_protos(model, dl, criterion, folder):
    """ Returns the test accuracy and loss.
    """
    loss, total, correct = 0.0, 0.0, 0.0

    agg_protos_label = {}
    model.to('cuda')  # 个人认为这里的划分无任何意义（不同本地的模型在本地测试集上的原型，之间的平均只能说明经验问题，也就是否都学得好）

    model.eval()
    for images, labels in dl:
        images, labels = images.to('cuda'), labels.to('cuda')

        model.zero_grad()
        outputs, protos = model(images, True)

        batch_loss = criterion(outputs, labels)
        loss += batch_loss.item()

        # prediction
        _, pred_labels = torch.max(outputs, 1)
        pred_labels = pred_labels.view(-1)
        correct += torch.sum(torch.eq(pred_labels, labels)).item()
        total += len(labels)

        for i in range(len(labels)):
            if labels[i].item() in agg_protos_label:
                agg_protos_label[labels[i].item()].append(protos[i, :])
            else:
                agg_protos_label[labels[i].item()] = [protos[i, :]]

    x = []
    y = []
    for label in agg_protos_label.keys():
        for proto in agg_protos_label[label]:
            tmp = proto.cpu().detach().numpy()
            x.append(tmp)
            y.append(label)

    x = np.array(x)
    y = np.array(y)
    np.save(folder + '_protos.npy', x)
    np.save(folder + '_labels.npy', y)

    print("Save protos and labels successfully.")