import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader
from src.plot import make_plot

Device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
GENERATE_PLOTS = False
# Test the KNN on the remaining, unseen, 20% of data
def test(dataset, train_subset, model_name, save_plot_path=None):
    print('Test begins here')
    train_subset = train_subset.tolist()
    test_data = []
    print('Checking if data was used during training')
    for x in dataset.full_data:
        sample = list(x)
        if sample not in train_subset:
            test_data.append(sample)

    print('Creating matrices for inputs and outputs')
    test_data_inputs = np.zeros([len(test_data), dataset.n_input_params], dtype='float32')
    test_data_outputs = np.zeros([len(test_data), dataset.n_output_params], dtype='float32')
    test_data = np.array(test_data)

    for i in range(len(test_data)):
        test_data_inputs[i] = test_data[i][range(0, dataset.n_input_params)]
        test_data_outputs[i] = test_data[i][range(dataset.n_input_params,
                                                  dataset.n_input_params + dataset.n_output_params)]

    inputs_test_tensor = torch.from_numpy(test_data_inputs)
    outputs_test_tensor = torch.from_numpy(test_data_outputs)
    test_data_set_1 = TensorDataset(inputs_test_tensor, outputs_test_tensor)
    test_data_loader_1 = DataLoader(test_data_set_1, batch_size=1, shuffle=False)

    mre_plot, y_pred_plot, y_true_plot = [], [], []
    sqr, sqt = [], []

    print('Loading model...')
    model = torch.load(model_name)
    model.eval()

    print('Evaluating model...')
    with torch.no_grad():
        for x, y in test_data_loader_1:
            y_pred = model(x)
            y_true = y
            tensor_rel_error = abs((y_pred - y_true) / y_true)
            mean_rel_error = torch.sum(tensor_rel_error) / tensor_rel_error.numel()

            mre_plot.append(mean_rel_error.item() * 100)
            y_pred_plot.append(y_pred.item())
            y_true_plot.append(y_true.item())

    # Metrics
    y_mean = sum(y_true_plot) / len(y_true_plot)
    sqr = [(y_true_plot[i] - y_pred_plot[i])**2 for i in range(len(y_true_plot))]
    sqt = [(y_true_plot[i] - y_mean)**2 for i in range(len(y_true_plot))]
    r2 = 1 - (sum(sqr) / sum(sqt))

    delta = [abs(y_pred_plot[i] - y_true_plot[i]) for i in range(len(y_true_plot))]
    nmae = (1 / len(y_true_plot)) * sum(delta) / y_mean

    correct = sum(1 for err in mre_plot if err <= 25)
    test_accuracy = (correct / len(y_true_plot)) * 100

    print(f"R²: {r2:.4f}")
    print(f"NMAE: {nmae:.4f}")
    print(f"P(error <= 25%): {test_accuracy:.2f}%")

    if GENERATE_PLOTS:
        make_plot(mre_plot, y_pred_plot, y_true_plot, save_path=save_plot_path)

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