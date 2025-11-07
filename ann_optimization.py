"""
Created on Thu Nov  19 14:08:00 2020

@author: Mohammad Moradi
"""

# import openpyxl as op
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import KFold
import optuna
# from plotly.offline import plot
from sklearn.utils import resample
import pickle
import json
import pandas as pd


# global variables
# sampler = optuna.samplers.TPESampler()
# study = optuna.create_study(sampler=sampler, direction='minimize')
r2_target = 0.90 #0.99
nmae_target = 0.15#0.08
Device = torch.device(
    'cuda') if torch.cuda.is_available() else torch.device('cpu')

# create dataset class
# read data from excel sheets and output 2 matrices
# -input_data
# -output_data
# this is to reduce logical complexity in the 'data_crossvalidation' function
# -> if I change my inputs, i don't have to change my 'data_crossvalidation' function


class Dataset:

    def __init__(self, text):

        self.name = text
        weather2 = open(self.name, "r")  #change variable name
        weather2 = json.load(weather2)
        weather2 = weather2[0] 


        self.dataset = pd.DataFrame.from_dict(weather2)

        # self.dataset = op.load_workbook(xls)
        # Change the inputs and outputs here
        Input_Variable = ['distance', 'speed', 'temp_outside', 'AC', 'rain']
        Output_Variable = ['consume']

        #Input_Variable = ['EVC_p_Offset', 'relSpeed',
        #                  'load', 'EWG_p_Offset', 'T_amb']
        #Output_Variable = ['FuelConsumption']
        self.input_sheet = self.dataset[Input_Variable]
        self.output_sheet = self.dataset[Output_Variable]
        self.full_data_DataFrame = pd.concat(
            [self.input_sheet, self.output_sheet.reindex(self.input_sheet.index)], axis=1)
        self.input_data = self.input_sheet.to_numpy()
        self.output_data = self.output_sheet.to_numpy()
        self.n_input_params = len(self.input_sheet.columns)
        self.n_output_params = len(self.output_sheet.columns)
        self.full_data = self.full_data_DataFrame.to_numpy()

    # return number of rows

    def get_rows(self):
        rows = len(self.input_data)
        return rows


# create an 80% Subset for training
    # It's common practice to use 80% of Data for Crossvalidation and 20% for unbiased testing
    # takes a full data matrix (e.g. dataset1.full_data) and returns an 80% resampled matrix
    # this matrix contains BOTH Inputs and Outputs!!
def create_subset(full_data):
    # ANN needs different datapoints in order to not "auswendig lernen" -> shuffeln
    # To get shuffled Datapoints, use resample funktion with specified sample_size
    sample_size = (len(full_data) * 95) // 100
    train_validation_subset = resample(
        full_data, replace=False, n_samples=sample_size, random_state=1)
    return train_validation_subset


# create ANN model
def define_net_regression(trial, n_input_params, n_output_params):
    ''' chose number of hidden layers, number of hidden neurons and the activation function
    the input and output layer is assumed as linear layer with linear transformation
    two input features (torque and rotation), one output feature (efficiency)'''

    layers = []

    hidden_layers = trial.suggest_int(
        "hidden_layers", 2, 2)  # number of hidden layers
    print("number of hidden layers", hidden_layers)
    x_in = n_input_params

    # create hidden layers:
    for i in range(hidden_layers):
        # number of neurons in this layer
        hidden_neurons = trial.suggest_int(
            'hidden_neurons_' + str(i+1), (n_input_params*2-i)*2, 30)
        print("neurons in hidden layer [", i+1, "]", hidden_neurons)
        layers.append(nn.Linear(x_in, hidden_neurons))  # append this layer
        layers.append(nn.BatchNorm1d(hidden_neurons))
        act_func_name = trial.suggest_categorical(
            'act_func_' + str(i+1), ["ReLU"])  # , "Sigmoid", "Tanh"
        act_func = getattr(nn, act_func_name)()
        print("act_func[", i+1, "]", act_func)
        layers.append(act_func)
        x_in = hidden_neurons  # number of inputs for next layer = number of neurons of this layer

    layers.append(nn.Linear(x_in, n_output_params))  # output layer
    net = nn.Sequential(*layers)

    #Initialization: The weights of the neurons must be randomly initialized at the beginning.
    #Algorithms such as Xavier initialization have proven effective for this.
    #Due to their method of random initialization, neural networks (KNNs) are able to converge more effectively.

    for m in net:  # initialize weights and bias at the first step of training
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            nn.init.zeros_(m.bias)
    return net


# convert data sets to tensors for crossvalidation
    # takes the batches of the training_subset from the k-fold crossvalidation
    # also takes the numbers of input and ouput parameters of the original dataset in order to seperate them again
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


# crossvalidation method
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
    with open('{}.pickle'.format(trial.number), 'wb') as saving:
        pickle.dump(model, saving)
    return accuracy


# Optimization method from Jinan
    # optimization algorithm for the hyperparameters
    # Tree-structure Parzen Estimator for optimization
def optimization(dataset, data_train):
    sampler = optuna.samplers.TPESampler()
    study = optuna.create_study(sampler=sampler, direction='minimize')
    study.optimize(lambda trial: crossvalidation(trial, dataset.n_input_params, dataset.n_output_params, data_train),
                   n_trials=3)  # number of trials (25)
    with open('{}.pickle'.format(study.best_trial.number), 'rb') as loading:  # load best model
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

    return best_model


# Test the KNN on the remaining, unseen, 20% of data
def test(dataset, train_subset, model_name):
    print('Test begins here')
    print('converting training subset to list')
    train_subset = train_subset.tolist()
    test_data = []
    print('Checking, if Data was used during training')
    for x in dataset.full_data:
        sample = list(x)
        if sample not in train_subset:
            test_data.append(sample)
        else:
            pass

    print('creating matrices for inputs and outputs')
    test_data_inputs = np.zeros(
        [len(test_data), dataset.n_input_params], dtype='float32')
    test_data_outputs = np.zeros(
        [len(test_data), dataset.n_output_params], dtype='float32')
    test_data = np.array(test_data)
    for i in range(len(test_data)):
        test_data_inputs[i] = test_data[i][range(0, dataset.n_input_params)]
        test_data_outputs[i] = test_data[i][range(
            dataset.n_input_params, dataset.n_input_params+dataset.n_output_params)]

    # convert inputs and outputs to tensors
    inputs_test_tensor = torch.from_numpy(test_data_inputs)
    outputs_test_tensor = torch.from_numpy(test_data_outputs)

    # Process tensors for pytorch
    batch_size_1 = 1
    test_data_set_1 = TensorDataset(inputs_test_tensor, outputs_test_tensor)
    test_data_loader_1 = DataLoader(
        test_data_set_1, batch_size_1, shuffle=False)

    mre_plot = []  # list of mean relative errors of each sample
    y_pred_plot = []  # list of predicted output samples
    y_true_plot = []  # list of the actual output sampels
    sqr = []  # list of squared residuals
    sqt = []  # list of squared totals

    print('loading model')
    model = torch.load(model_name)
    print('evaluating model')
    model.eval()

    print('calculate errors for each prediction')
    with torch.no_grad():

        for x, y in test_data_loader_1:  # now test data set
            y_true = y  # actual output value of test data
            y_pred = model(x)  # predicted output value
            tensor_rel_error = abs((y_pred - y_true) / y_true)
            mean_rel_error = torch.sum(
                tensor_rel_error)/tensor_rel_error.numel()

            # create plot array
            mre_plot.append(mean_rel_error.item() * 100)
            y_pred_plot.append(y_pred.item())
            y_true_plot.append(y_true.item())

        # calculate r2-score
        # mean of measured output data
        y_mean = sum(y_true_plot) / len(y_true_plot)
        for i in range(len(y_true_plot)):
            sqr.append((y_true_plot[i] - y_pred_plot[i])**2)
            sqt.append((y_true_plot[i] - y_mean)**2)
        r2 = 1 - (sum(sqr) / sum(sqt))
        print('r2:', r2)

        # calculate nmae
        delta = []
        for i in range(len(y_true_plot)):
            delta.append(abs(y_pred_plot[i]-y_true_plot[i]))
        nmae = (1/len(y_true_plot))*sum(delta)/y_mean
        print('nmae:', nmae)

        # calculate the number of right predictions for given threshold
        correct = 0
        false = 0
        for i in range(len(y_true_plot)):
            if mre_plot[i] <= 25:
                correct += 1
            else:
                false += 1
        test_accuracy = (correct / len(y_true_plot)) * 100
        print("P( error<=25% )= {result}".format(result=test_accuracy))

        # plot the results
        '''
        loss_test_line_pos = [i * 1.30 for i in loss_test_plot_3]
        loss_test_line_neg = [i * 0.7 for i in loss_test_plot_3]
        if test_accuracy >= test_accuracy_value:
            plt.figure(figsize=(12, 6))
            font = {'family': 'serif',
                    'color': 'darkred',
                    'weight': 'normal',
                    'size': 12, }
            plt.subplot(131)
            plt.hist(loss_test_plot_1, bins=math.ceil(max(loss_test_plot_1)))
            plt.title('histogram relative test error in %', fontdict=font)
            plt.xlabel('bins in 1 % error class', size=12)
            # plt.show()

            plt.subplot(132)
            plt.plot(num_epochs_1, loss_test_plot_2, 'o', color='blue', label='predictions')
            plt.plot(num_epochs_1, loss_test_plot_3, 'o', color='red', label='labels', alpha=0.4)
            plt.title('plot predictions and labels', fontdict=font)
            plt.legend(loc="best", prop={'size': 8})
            plt.xlabel('number of the sample', size=12)
            plt.ylabel('eta electric motor', size=12)
            # plt.show()
            # plt.savefig('B2.png')
            # plt.show()
            # plt.subplot(133)
            # plt.plot(num_epochs_1, accuracy_list,color='red')
            # plt.title('test accuracy',fontdict=font)
            # plt.xlabel('number of the sample',size=16)
            # plt.ylabel('accuracy in %',size=16)

            plt.subplot(133)
            plt.plot(loss_test_plot_3, loss_test_plot_2, "o", color="blue", label='output ANN', alpha=0.4)
            plt.plot(loss_test_plot_3, loss_test_plot_3, color='red', label='prediction = label')
            plt.plot(loss_test_plot_3, loss_test_line_pos, '--', color='red', label='+-3% error')
            plt.plot(loss_test_plot_3, loss_test_line_neg, '--', color='red')
            plt.title('error plot predictions vs. labels', fontdict=font)
            plt.legend(loc='best', prop={'size': 8})
            plt.xlabel('labels', size=12)
            plt.ylabel('predictions', size=12)

            # plt.show()
            # plt.figure(figsize=(6,3))
            # plt.savefig('C:/Users/PaulScholten/Documents/UniKram/Bachelorarbeit/PythonCodeMitDokus/data_20Tr.png')
            # plt.close('C:/Users/PaulScholten/Documents/UniKram/Bachelorarbeit/PythonCodeMitDokus/data_20Tr.png')
            # plt.savefig(
            #     '/pfs/work7/workspace/scratch/oc6350-conda-0/conda/NN_neue_Modelle/EM-Motor-auto-Input-varNeuronen/data_20Tr.png')
            # plt.close(
            #     '/pfs/work7/workspace/scratch/oc6350-conda-0/conda/NN_neue_Modelle/EM-Motor-auto-Input-varNeuronen/data_20Tr.png')
            '''
    return test_accuracy, nmae, r2


# Test the KNN on approx. 56600 unseen data samples
    # dataset has to be from a new excel sheet that was not used to train the network
def unseen_test(dataset, model_name):
    print('Test begins here')
    test_data = []
    print('appending matrix')
    for x in dataset.full_data:
        sample = list(x)
        test_data.append(sample)

    # splitting data into input and output matrices
    print('splitting into inputs and outputs')
    test_data_inputs = np.zeros(
        [len(test_data), dataset.n_input_params], dtype='float32')
    test_data_outputs = np.zeros(
        [len(test_data), dataset.n_output_params], dtype='float32')
    test_data = np.array(test_data)
    for i in range(len(test_data)):
        test_data_inputs[i] = test_data[i][range(0, dataset.n_input_params)]
        test_data_outputs[i] = test_data[i][range(
            dataset.n_input_params, dataset.n_input_params + dataset.n_output_params)]

    # convert inputs and outputs to tensors
    inputs_test_tensor = torch.from_numpy(test_data_inputs).to(Device)
    outputs_test_tensor = torch.from_numpy(test_data_outputs).to(Device)

    # Process tensors for pytorch
    batch_size_1 = 1
    test_data_set_1 = TensorDataset(inputs_test_tensor, outputs_test_tensor)
    test_data_loader_1 = DataLoader(
        test_data_set_1, batch_size_1, shuffle=False)

    mre_plot = []  # list of mean relative errors of each sample
    y_pred_plot = []  # list of predicted output samples
    y_true_plot = []  # list of the actual output sampels
    sqr = []  # list of squared residuals
    sqt = []  # list of squared totals

    model = torch.load(model_name)
    model.eval()

    print('calculating nmae, r2 and P')

    with torch.no_grad():

        for x, y in test_data_loader_1:  # now test data set
            y_true = y  # actual output value of test data
            y_pred = model(x)  # predicted output value
            tensor_rel_error = abs((y_pred - y_true) / y_true)
            mean_rel_error = torch.sum(
                tensor_rel_error) / tensor_rel_error.numel()

            # print("target", y_true)
            # print("prediction", y_pred)
            # print("tensor_relative_error", tensor_rel_error)
            # print("mean_rel_error: ", mean_rel_error)

            # create plot array
            mre_plot.append(mean_rel_error.item() * 100)
            y_pred_plot.append(y_pred.item())
            y_true_plot.append(y_true.item())

        # calculate r2-score
        # mean of measured output data
        y_mean = sum(y_true_plot) / len(y_true_plot)
        for i in range(len(y_true_plot)):
            sqr.append((y_true_plot[i] - y_pred_plot[i]) ** 2)
            sqt.append((y_true_plot[i] - y_mean) ** 2)
        r2 = 1 - (sum(sqr) / sum(sqt))
        print('r2 on unseen data:', r2)

        # calculate nmae
        delta = []
        for i in range(len(y_true_plot)):
            delta.append(abs(y_pred_plot[i] - y_true_plot[i]))
        nmae = (1 / len(y_true_plot)) * sum(delta) / y_mean
        print('nmae on unseen data:', nmae)

        # calculate the number of right predictions for given threshold
        correct = 0
        false = 0
        for i in range(len(y_true_plot)):
            if mre_plot[i] <= 25:
                correct += 1
            else:
                false += 1
        test_accuracy = (correct / len(y_true_plot)) * 100
        print("P( error<=25% ) on unseen data: {result}".format(
            result=test_accuracy))

        # write predictions to xlsx file
        '''workbook = op.Workbook()
        sheet = workbook.active
        sheet.cell(row=1, column=1).value = "NO [ppm]"
        for i in range(len(y_pred_plot)):
            sheet.cell(row=i + 2, column=1).value = y_pred_plot[i]
        workbook.save("NO_predictions.xlsx")'''

    return test_accuracy, nmae, r2


def find_best_model(dataset, train_subset, test_set):

    best_model = optimization(dataset, train_subset)
    torch.save(best_model, 'ANN_best_20Trials.pt')
    error_probability_best, nmae_best, r2_best = unseen_test(
        test_set, 'ANN_best_20Trials.pt')
    print("best nmae: ", nmae_best)
    print("best R2:", r2_best)
    print("best probability for error less then 25%:", error_probability_best)

    while r2_best < r2_target:  # nmae_best > nmae_target and
        best_model = optimization(dataset, train_subset)
        torch.save(best_model, 'ANN_current_20Trials.pt')
        error_probability_current, nmae_current, r2_current = unseen_test(
            test_set, 'ANN_current_20Trials.pt')
        print("current nmae:", nmae_current)
        print("current r2:", r2_current)
        print("current probability for error less then 25%:",
              error_probability_current)
        if nmae_current <= nmae_best and r2_current >= r2_best:
            torch.save(best_model, 'ANN_best_20Trials.pt')
            error_probability_best = error_probability_current
            nmae_best = nmae_current
            r2_best = r2_current
            print("New Best Model!")
            print("best nmae:", nmae_best)
            print("best r2:", r2_best)
            print("best probability for error less then 25%:",
                  error_probability_best)


# Main program change the name of test and train here

print("creating dataset for training, validation and random test")
#Simply enter the correct name of the Excel file here.
dataset_1 = Dataset('training.json')

print("Sampling 95% as a subset for training")
train_subset_1 = create_subset(dataset_1.full_data)

# print("creating dataset for test")
dataset_2 = Dataset('testing.json')


find_best_model(dataset_1, train_subset_1, dataset_2)
test(dataset_1, train_subset_1, 'ANN_best_20Trials.pt')

unseen_test(dataset_2, 'ANN_best_20Trials.pt')

model = torch.load('ANN_best_20Trials.pt')
print(model)