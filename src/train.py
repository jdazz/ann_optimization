import torch
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import KFold
import optuna
import numpy as np
import pickle, os
from src.model import define_net_regression
from .model_test import unseen_test

Device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def data_crossvalidation(n_input_params, n_output_params, train, validate, batch_size):
    # next, split the Subset back into Inputs and Outputs
    input_train = np.zeros([len(train), n_input_params], dtype='float32')
    output_train = np.zeros([len(train), n_output_params], dtype='float32')
    input_validate = np.zeros([len(validate), n_input_params], dtype='float32')
    output_validate = np.zeros(
        [len(validate), n_output_params], dtype='float32')

    for i in range(len(train)):
        for j in range(n_input_params):
            input_train[i][j] = train[i][j]
        for k in range(n_output_params):
            output_train[i][k] = train[i][k + n_input_params]

    for i in range(len(validate)):
        for j in range(n_input_params):
            input_validate[i][j] = validate[i][j]
        for k in range(n_output_params):
            output_validate[i][k] = validate[i][k + n_input_params]

    # Convert to Tensors
    input_train = torch.from_numpy(input_train).to(Device)
    output_train = torch.from_numpy(output_train).to(Device)
    input_validate = torch.from_numpy(input_validate).to(Device)
    output_validate = torch.from_numpy(output_validate).to(Device)

    train_data_set = TensorDataset(input_train, output_train)
    train_data_loader = DataLoader(
        train_data_set, batch_size, shuffle=True, drop_last=True)
    validation_data_set = TensorDataset(input_validate, output_validate)
    validation_data_loader = DataLoader(
        validation_data_set, batch_size, shuffle=False)

    return train_data_loader, validation_data_loader, output_validate

def crossvalidation(trial, n_input_params, n_output_params, data_train):
    ''' repeated crossvalidation of the model
        optimization of the following parameters (learning rate, batch size, epochs and optimizer weights)'''
    # create model and send it to cpu or gpu
    print("creating net")
    model = define_net_regression(
        trial, n_input_params, n_output_params).to(Device)

    # loss function will spit out a single number as to quantify, how bad the prediction is
    # define the loss function to be the mean square root
    loss_function = F.mse_loss

    learning_rate = trial.suggest_loguniform("learning_rate", 0.0001, 0.01)
    batch_size = trial.suggest_int("batch_size", 50, 150)
    epochs = trial.suggest_int("epochs", 200, 200)
    optimizer_name = trial.suggest_categorical(
        "opt", ["Adam"])  # , "RMSprop", "SGD"
    opt = getattr(torch.optim, optimizer_name)(
        model.parameters(), lr=learning_rate)

    k = 2  # 8-fold crossvalidation
    print('starting training with', epochs, 'epochs')
    model.train()
    for epoch in range(epochs):
        # print('Epoch', epoch+1)
        # , True)  # every new 8-fold, the training dataset is randomly shuffled
        kfold = KFold(k)
        for train, validate in kfold.split(data_train):
            train = data_train[train]
            validate = data_train[validate]

            train_data_loader, validation_data_loader, output_validate = data_crossvalidation(n_input_params,
                                                                                              n_output_params,
                                                                                              train, validate,
                                                                                              batch_size)

            # perform stochastic gradient descent over one batch
            # print('performing backward propagation')
            for x, y in train_data_loader:
                # x is one row ( or "sample") of inputs
                # y is one row (or "sample") of outputs
                # x, y = x.to(Device), y.to(Device)
                # model(x) will now return a predicted row of outputs
                pred = model(x)
                # calculate MSE over all output parameters
                loss = loss_function(pred, y)
                opt.zero_grad()  # reset gradients to zero
                loss.backward()  # compute gradients
                opt.step()  # updates parameters of the net

            # print('evaluation model')
            model.eval()
            with torch.no_grad():
                for x, y in validation_data_loader:  # now test data set
                    # x, y = x.to(Device), y.to(Device)
                    pred_validate = model(x)
                    accuracy = loss_function(pred_validate, y)

        # print('tracking the accuracy of this trial and prune if necessary')
        # track the accuracy for the optimization
        trial.report(accuracy, epoch)
        if trial.should_prune():
            raise optuna.exceptions.TrialPruned()

    # save the net (parameters, structure)

    with open(os.path.join("models", f"trial_{trial.number}.pickle"), "wb") as saving:
        pickle.dump(model, saving)
    return accuracy

# Optimization method from Jinan
    # optimization algorithm for the hyperparameters
    # Tree-structure Parzen Estimator for optimization
def optimization(dataset, data_train):
    sampler = optuna.samplers.TPESampler()
    study = optuna.create_study(sampler=sampler, direction='minimize')
    study.optimize(lambda trial: crossvalidation(trial, dataset.n_input_params, dataset.n_output_params, data_train),
                   n_trials=30)  # number of trials (25)
  
    best_model_path = os.path.join("models", f"trial_{study.best_trial.number}.pickle")  # load best model
    with open(best_model_path, "rb") as loading:
        best_model = pickle.load(loading)

    pruned_trials = [t for t in study.trials if t.state ==
                     optuna.trial.TrialState.PRUNED]
    complete_trials = [t for t in study.trials if t.state ==
                       optuna.trial.TrialState.COMPLETE]

    print("Study statistics: ")
    print("  Number of finished trials: ", len(study.trials))
    print("  Number of pruned trials: ", len(pruned_trials))
    print("  Number of complete trials: ", len(complete_trials))

    print("Best trial:")
    trial = study.best_trial

    print("  Value: ", trial.value)

    print("  Params: ")
    for key, value in trial.params.items():
        print("    {}: {}".format(key, value))

    print(study.best_params)
    # cleanup intermediate trial models
    for file in os.listdir("models"):
        if file.startswith("trial_") and not file.endswith(f"{study.best_trial.number}.pickle"):
            os.remove(os.path.join("models", file))

    return best_model

def find_best_model(dataset, train_subset, test_set):

    print("Running optimization (3 trials)...")
    best_model = optimization(dataset, train_subset)
    
    print("Saving best model...")
    best_model_path = os.path.join("models", "ANN_best_20Trials.pt")
    torch.save(best_model, best_model_path)
    
    print("Testing best model on unseen data...")
    error_probability_best, nmae_best, r2_best = unseen_test(
        test_set, best_model_path)
        
    print(f"Final Best NMAE: {nmae_best}")
    print(f"Final Best R2: {r2_best}")
    print(f"Final Best P(error<25%): {error_probability_best}")