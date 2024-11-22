#Standart Libs imports:
import torch
from torch.utils.data import DataLoader
import os
import numpy as np
from tqdm import tqdm
from datetime import date
import pickle
import json
import copy
import torch.nn as nn

#Custom imports:
from data import Videos



def setup_dir(
    output_dir,
    train_desc=""
    ):
    try:
        os.mkdir(output_dir)
    except FileExistsError:
        pass
    current_date = date.today()
    name = current_date.strftime("%d-%m-%y")
    training_output_dir = os.path.join(output_dir, name+train_desc).replace("\\","/")
    os.mkdir(training_output_dir)
    return training_output_dir

class EarlyStopper():
    def __init__(self, patience=1, min_delta=0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.min_validation_loss = float('inf')

    def early_stop(self, validation_loss):
        if validation_loss < self.min_validation_loss:
            self.min_validation_loss = validation_loss
            self.counter = 0
        elif validation_loss > (self.min_validation_loss + self.min_delta):
            self.counter += 1
            if self.counter >= self.patience:
                return True
        return False

def save_evolution(model, optimizer, history, out_dir, epoch):
    history_path = os.path.join(out_dir, f"metrics.p").replace("\\","/")
    model_state_path = os.path.join(out_dir, f"model_state_{epoch}.pt").replace("\\","/")
    optim_state_path = os.path.join(out_dir, f"optim_state_{epoch}.pt").replace("\\","/")
    torch.save(model.state_dict(), model_state_path)
    torch.save(optimizer.state_dict(), optim_state_path)
    with open(history_path, 'wb') as fp:
        pickle.dump(
            history, 
            fp, 
            protocol=pickle.HIGHEST_PROTOCOL
        )

def log_behaviour(json_path, metrics_dict):
    with open(json_path, "a+") as f:
        json.dump(metrics_dict, f, indent=4)

def train_epoch(batch, label, model, device, loss_function, optimizer, log_path, epoch):
    optimizer.zero_grad()
    batch = batch.to(device)
    label = label.to(device)
    model_output = model(batch)
    if isinstance(loss_function, nn.GaussianNLLLoss):
        num_outputs = model_output.size(1) // 2
        prediction, log_variance = model_output[:, :num_outputs], model_output[:, num_outputs:]
        variance = torch.exp(log_variance)
        batch_loss = loss_function(prediction, label.float(), variance)
        if log_path:
            log_behaviour(
                json_path = log_path,
                metrics_dict = {
                    f"Epoch {epoch}" : {
                        "prediction" : {
                            "median" : torch.median(prediction).detach().to("cpu").tolist(), 
                            "prediction" : torch.mean(prediction).detach().to("cpu").tolist(), 
                            "std" : torch.std(prediction).detach().to("cpu").tolist(), 
                            "max" : torch.max(prediction).detach().to("cpu").tolist(), 
                            "min" : torch.min(prediction).detach().to("cpu").tolist()
                        },
                        "log_variance" : {
                            "median" : torch.median(log_variance).detach().to("cpu").tolist(), 
                            "prediction" : torch.mean(log_variance).detach().to("cpu").tolist(), 
                            "std" : torch.std(log_variance).detach().to("cpu").tolist(), 
                            "max" : torch.max(log_variance).detach().to("cpu").tolist(), 
                            "min" : torch.min(log_variance).detach().to("cpu").tolist()
                        },
                        "variance" : {
                            "median" : torch.median(variance).detach().to("cpu").tolist(), 
                            "prediction" : torch.mean(variance).detach().to("cpu").tolist(), 
                            "std" : torch.std(variance).detach().to("cpu").tolist(), 
                            "max" : torch.max(variance).detach().to("cpu").tolist(), 
                            "min" : torch.min(variance).detach().to("cpu").tolist()
                        }
                    }
                }
            )
    else:
        batch_loss = loss_function(prediction, label.float())
    batch_loss.backward()
    optimizer.step()
    return batch_loss

def evaluate_model(batch, label, model, device, loss_function):
    batch = batch.to(device)
    label = label.to(device)
    model_output = model(batch)
    if isinstance(loss_function, nn.GaussianNLLLoss):
        num_outputs = model_output.size(1) // 2
        prediction, log_variance = model_output[:, :num_outputs], model_output[:, num_outputs:]
        variance = torch.exp(log_variance)
        batch_loss = loss_function(prediction, label, variance)
    else:
        batch_loss = loss_function(prediction, label)
    return batch_loss

def activate_dropout(model):
    for module in model.modules():
        if isinstance(module, nn.Dropout):
            module.train()

def train(
    model,
    n_epochs,
    train_loader,
    valid_loader,
    loss_fn,
    lr,
    output_dir,
    patience,
    min_delta,
    save_every=5,
    train_desc="",
    device=None,
    mc_dropout=False,
    log=False
    ):
    epoch_pbar = tqdm(
        total = n_epochs,
        position = 1, 
        desc = "Epoch       ",
        leave = False
    )
    batch_train_pbar = tqdm(
        total = len(train_loader),
        position = 2, 
        desc = "Train       ",
        leave = False
    )
    batch_valid_pbar = tqdm(
        total = len(valid_loader),
        position = 3, 
        desc = "Validation  ",
        leave = False
    )
    out_dir = setup_dir(output_dir, train_desc=train_desc)
    model = model.to(device)
    optimizer = torch.optim.Adam(
        model.parameters(), 
        lr=lr
        )
    history = {}
    early_stopper = EarlyStopper(
        patience=patience, 
        min_delta=min_delta
        )
    if log:
        log_path = os.path.join(out_dir, "train_log.json").replace("\\","/")
    else:
        log_path = None
    for epoch in range(n_epochs):
        model.train()
        train_loss = 0
        for i, (batch, label, _) in enumerate(train_loader):
            batch_loss = train_epoch(
                batch = batch,
                label = label,
                model = model,
                device = device,
                loss_function = loss_fn,
                optimizer = optimizer,
                log_path = log_path
            )
            train_loss += batch_loss
            batch_train_pbar.update(1)
        batch_train_pbar.reset()
        avg_train_loss = train_loss / i
        model.eval()
        if mc_dropout:
            activate_dropout(model)
        valid_loss = 0
        with torch.no_grad():
            for j, (batch, label, _) in enumerate(valid_loader):
                batch_loss = evaluate_model(
                    batch = batch,
                    label = label,
                    model = model,
                    device = device,
                    loss_function = loss_fn
                )
                valid_loss += batch_loss
                batch_valid_pbar.update(1)
        batch_valid_pbar.reset()       
        avg_valid_loss = valid_loss / len(valid_loader)        
        history[f"{epoch}"] = {
            "Train Loss": float(avg_train_loss.detach().to("cpu").numpy()),
            "Valid Loss": float(avg_valid_loss.detach().to("cpu").numpy())
        }
        epoch_pbar.update(1)
        if epoch % save_every == 0:
            save_evolution(
                model = model,
                optimizer = optimizer,
                history = history,
                out_dir = out_dir,
                epoch = epoch
                )
            last_saved_epoch = epoch
        if avg_valid_loss < early_stopper.min_validation_loss:
            save_evolution(
                model = model,
                optimizer = optimizer,
                history = history,
                out_dir = out_dir,
                epoch = "best"
                )
            best_loss = {
                "Train Loss" : history[f"{epoch}"]["Train Loss"],
                "Valid Loss" : history[f"{epoch}"]["Valid Loss"]
            }
        if early_stopper.early_stop(avg_valid_loss):
            print("Early Stop condition met.")             
            break
    last_loss = {
        "Train Loss" : history[f"{epoch}"]["Train Loss"],
        "Valid Loss" : history[f"{epoch}"]["Valid Loss"]
    }
    return history, best_loss, last_loss, out_dir, last_saved_epoch

class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NpEncoder, self).default(obj)
    
def test_model(batch, label, model, device, loss_function):
    batch = batch.to(device)
    label = label.to(device)
    model_output = model(batch)
    if isinstance(loss_function, nn.GaussianNLLLoss):
        num_outputs = model_output.size(1) // 2
        prediction, log_variance = model_output[:, :num_outputs], model_output[:, num_outputs:]
        variance = torch.exp(log_variance)
        batch_loss = loss_function(prediction, label, variance)
    else:
        batch_loss = loss_function(prediction, label)
    return batch_loss

def test(dataloader, loss_fn, device, model, pbar):  
    test_loss = 0
    for batch, labels, _ in dataloader:
        batch_loss = test_model(
            batch = batch,
            label = labels,
            model = model,
            device = device,
            loss_function = loss_fn
        )
        test_loss += batch_loss
        pbar.update(1)
    test_loss *= 1/len(dataloader)
    return test_loss

def get_loader_idxs(dataloader):
    idx_set = set()
    for _, _, idxs in dataloader:
        idxs = idxs.detach().to("cpu").numpy()
        i = set(idxs)
        idx_set = idx_set | i
    return idx_set

def run_multiple_trains(
    configs_list : list,
    train_dir : str,
    default_config : dict,
    return_test_idxs = False,
    device=None
    ):
    
    current_date = date.today()
    name = current_date.strftime("%d-%m-%y")
    output_dir = os.path.join(train_dir, name).replace("\\", "/")
    summary_path = os.path.join(output_dir, "summary.json").replace("\\", "/")
    store_configs_path = os.path.join(output_dir, "configs.json").replace("\\", "/")
    setup_dir(output_dir)
    
    summary = {}
    config_pbar = tqdm(
        total = len(configs_list),
        position = 0, 
        desc = "Config      "
    )
    for n_config, config in enumerate(configs_list):
        current_trains_desc = f"_config_{n_config}"
        current_train_config = default_config
        for key in config.keys():
            current_train_config[key] = config[key]
        
        _config = copy.deepcopy(current_train_config)
        for key in ["loss_fn", "model"]:
            _config[key] = str(_config[key])
        with open(store_configs_path, "w+") as f:
            json.dump(_config, f, indent=4)
    
        data = Videos(
            videos_dir = current_train_config["data_path"],
            generation_df_path = current_train_config["generation_df_path"],
            rescale_labels = current_train_config["rescale_labels"],
            labels = current_train_config["param"],
            no_td = current_train_config["no_td"],
            normalize_data = current_train_config["normalize_data"]
        )
        
        g_cpu = torch.Generator()
        g_cpu.manual_seed(2147483647)
        total_train_set, total_valid_set, total_test_set = torch.utils.data.random_split(data, [0.7,0.2,0.1], generator=g_cpu)
        ratio = current_train_config["data_size"] / len(total_train_set)
        _, train_set = torch.utils.data.random_split(total_train_set, [1-ratio,ratio], generator=g_cpu)
        _, valid_set = torch.utils.data.random_split(total_valid_set, [1-ratio,ratio], generator=g_cpu) # [!] Affects thesis results.
        _, test_set = torch.utils.data.random_split(total_test_set, [1-ratio,ratio], generator=g_cpu)   # [!] Affects thesis results.
        
        train_dataloader = DataLoader(train_set, batch_size=current_train_config["batch_size"], shuffle=True)
        valid_dataloader = DataLoader(valid_set, batch_size=current_train_config["batch_size"], shuffle=True)
        test_dataloader  = DataLoader(test_set,  batch_size=current_train_config["batch_size"], shuffle=True)
        if return_test_idxs:
            test_idxs_path = os.path.join(output_dir, "test_idxs.json").replace("\\", "/")
            idxs = list(get_loader_idxs(test_dataloader))
            with open(test_idxs_path, 'w') as file:
                json.dump(idxs, file, cls=NpEncoder)
        
        history, best_loss, last_loss, out_dir, last_epoch = train(
            model = current_train_config["model"],
            n_epochs = current_train_config["epochs"],
            train_loader = train_dataloader,
            valid_loader = valid_dataloader,
            loss_fn = current_train_config["loss_fn"],
            lr = current_train_config["lr"],
            save_every = 500,
            output_dir = output_dir,
            train_desc = current_trains_desc,
            patience = current_train_config["patience"],
            min_delta = current_train_config["min_delta"],
            device=device,
            mc_dropout = current_train_config["mc_dropout"],
            log = current_train_config["log_train"]
        )
        
        with torch.no_grad():
            for state, state_loss in zip(["best", f"{last_epoch}"],[best_loss, last_loss]):
                test_pbar = tqdm(
                    total = len(test_dataloader),
                    position = 5, 
                    desc = "Testing     ",
                    leave = False
                )
                model_state_path = os.path.join(out_dir, f"model_state_{state}.pt").replace("\\", "/")
                trained_model = current_train_config["model"]
                trained_model.load_state_dict(torch.load(model_state_path))
                trained_model.to(device)
                trained_model.eval()
                test_loss = test(
                    dataloader = test_dataloader,
                    loss_fn = current_train_config["loss_fn"],
                    device = device,
                    model = trained_model,
                    pbar = test_pbar
                )
                state_loss["Test Loss"] = float(test_loss.detach().to("cpu").numpy())
        
        summary[current_trains_desc] = {
            "best" : best_loss,
            "last" : last_loss
        }
        
        with open(summary_path, "w+") as f:
            json.dump(summary, f, indent=4)
        config_pbar.update(1)
        
        
