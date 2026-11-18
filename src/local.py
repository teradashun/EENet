import torch
import torch.nn as nn
import torch.nn.functional as F
from src.loss_functions import loss

def train(optimizer, model, train_loader, device, model_name, loss_func, frozen_prefixes=()):
    model.train()

    # 共有層のBatchNorm統計も固定する。
    for name, module in model.named_modules():
        if name and (name + ".").startswith(frozen_prefixes):
            module.eval()

    for i, (images, labels) in enumerate(train_loader):
        if model_name in ["DNN"]:
            images, labels = images.view(-1, 28*28).to(device), labels.to(device)
        
        else:
            images, labels = images.to(device), labels.to(device)
        
        optimizer.zero_grad()

        preds = model(images)
        pred_loss = loss(loss_func, device, preds, labels)

        pred_loss.backward()
        # torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()


def test(model, test_loader, device, model_name):
    model.eval()

    with torch.no_grad():
        correct_preds = 0
        total_preds = 0

        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)

            pred = model(inputs)
            
            _, outputs = torch.max(pred, dim=1)

            correct_preds += outputs.eq(labels).sum().item()
            total_preds += outputs.size(0)

        acc = 100*correct_preds/total_preds
    
    return acc


def early_test(model, test_loader, device, threshold, model_name, is_exit):
    model.eval()

    with torch.no_grad():
        correct_preds = 0
        total_preds = 0

        for dummy_inputs, _ in test_loader:
            dummy_inputs = dummy_inputs.to(device)
            if is_exit:
                _ = model(dummy_inputs, threshold)
            else:
                _ = model(dummy_inputs)
            break

        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)

            if is_exit:
                pred, _ = model(inputs, threshold)
            else:
                pred = model(inputs)
            
            _, outputs = torch.max(pred, dim=1)

            correct_preds += outputs.eq(labels).sum().item()
            total_preds += outputs.size(0)

        acc = 100*correct_preds/total_preds
    
    return acc


def fine_tune_personal(
    optimizer, model, train_loader, device, model_name,
    shared_prefixes, loss_func, epochs=1,
):
    original_flags = {
        name: p.requires_grad
        for name, p in model.named_parameters()
    }

    try:
        for name, p in model.named_parameters():
            if name.startswith(shared_prefixes):
                p.requires_grad_(False)

        for _ in range(epochs):
            train(
                optimizer, model, train_loader, device,
                model_name, loss_func,
                frozen_prefixes=shared_prefixes,
            )
    finally:
        for name, p in model.named_parameters():
            p.requires_grad_(original_flags[name])