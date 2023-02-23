import torch
import numpy as np
import torch.nn.functional as F

def trainer_loader():
    return train

def initialize_weight(model,loader):
    batch = next(iter(loader))
    device = next(model.parameters()).device
    with torch.no_grad():
        model(batch[0].to(device))

def train(model,optpack,train_loader,test_loader,print_steps=-1,log_results=False,log_path='log.txt'):
    model.train()
    opt = optpack["optimizer"](model.parameters())
    if optpack["scheduler"] is not None:
        sched = optpack["scheduler"](opt)
    else:
        sched = None
    num_steps = optpack["steps"]
    device = next(model.parameters()).device
    
    results_log = []
    training_step = 0
    
    if sched is not None:
        while True:
            for i,(x,y) in enumerate(train_loader):
                training_step += 1
                x = x.to(device)
                y = y.to(device)
        
                opt.zero_grad()
                yhat = model(x)
                loss = F.cross_entropy(yhat,y)
                loss.backward()
                opt.step()
                sched.step()
                
                if print_steps != -1 and training_step%print_steps == 0:
                    train_acc,train_loss    = test(model,train_loader)
                    test_acc,test_loss      = test(model,test_loader)
                    print(f'Steps: {training_step}/{num_steps} \t Train acc: {train_acc:.2f} \t Test acc: {test_acc:.2f}', end='\r')
                    if log_results:
                        results_log.append([test_acc,test_loss,train_acc,train_loss])
                        np.savetxt(log_path,results_log)
                if training_step >= num_steps:
                    break
            if training_step >= num_steps:
                break
    else:
        while True:
            for i,(x,y) in enumerate(train_loader):
                training_step += 1
                x = x.to(device)
                y = y.to(device)
        
                opt.zero_grad()
                yhat = model(x)
                loss = F.cross_entropy(yhat,y)
                loss.backward()
                opt.step()
                
                if print_steps != -1 and training_step%print_steps == 0:
                    train_acc,train_loss    = test(model,train_loader)
                    test_acc,test_loss      = test(model,test_loader)
                    print(f'Steps: {training_step}/{num_steps} \t Train acc: {train_acc:.2f} \t Test acc: {test_acc:.2f}', end='\r')
                    if log_results:
                        results_log.append([test_acc,test_loss,train_acc,train_loss])
                        np.savetxt(log_path,results_log)
                if training_step >= num_steps:
                    break
            if training_step >= num_steps:
                break
    train_acc,train_loss    = test(model,train_loader)
    test_acc,test_loss      = test(model,test_loader)
    print(f'Train acc: {train_acc:.2f}\t Test acc: {test_acc:.2f}')
    return [test_acc,test_loss,train_acc,train_loss]

def test(model,loader):
    model.eval()
    device = next(model.parameters()).device
    
    correct = 0
    loss    = 0
    total   = 0
    for i,(x,y) in enumerate(loader):
        x = x.to(device)
        y = y.to(device)
        with torch.no_grad():
            yhat    = model(x)
            _,pred  = yhat.max(1)
        correct += pred.eq(y).sum().item()
        loss += F.cross_entropy(yhat,y)*len(x)
        total += len(x)
    acc     = correct/total * 100.0
    loss    = loss/total
    
    model.train()
    
    return acc,loss