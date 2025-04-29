import torch
import torchmetrics
from evaluation import evaluate_accuracy_next_activity, logic_loss, logic_loss_multiple_samples
import sys
if torch.cuda.is_available():
     device = 'cuda:0'
else:
    device = 'cpu'
from statistics import mean

def train(rnn, train_dataset, test_dataset, max_num_epochs, epsilon, deepdfa=None, prefix_len=0, batch_size=64, logic_loss_type="one_sample"):
    if deepdfa is None:
        raise ValueError("deepdfa must be provided for logic-only training")
    
    curr_temp = 0.5
    lambda_temp = 0.9999999999
    min_temp = 0.0001
    rnn = rnn.to(device)
    optim = torch.optim.Adam(params=rnn.parameters(), lr=0.0005)
    acc_func = torchmetrics.Accuracy(task="multiclass", num_classes=train_dataset.size()[-1], top_k=1).to(device)
    old_loss = 1000

    ############################ TRAINING
    for epoch in range(max_num_epochs*2):
        current_index = 0
        train_acc_batches = []
        log_loss_batches = []
        
        while current_index <= train_dataset.size()[0]:
            initial = current_index
            final = min(current_index + batch_size, train_dataset.size()[0] + 1)
            current_index = final
            
            X = train_dataset[initial:final, :-1, :].to(device)
            Y = train_dataset[initial:final, 1:, :]
            target = torch.argmax(Y.reshape(-1, Y.size()[-1]), dim=-1).to(device)

            optim.zero_grad()
            
            predictions, _ = rnn(X)
            predictions = predictions.reshape(-1, predictions.size()[-1])
            
            ##################################### ONLY logic loss
            if logic_loss_type == "multiple_samples":
                log_loss = logic_loss_multiple_samples(rnn, deepdfa, X, prefix_len, curr_temp, num_samples=10)
            else:  # one_sample
                log_loss = logic_loss(rnn, deepdfa, X, prefix_len, curr_temp)
            
            log_loss_batches.append(log_loss.item())
            loss = log_loss  # Only use logic loss

            loss.backward()
            optim.step()

            # Calculate accuracy for monitoring purposes
            train_acc_batches.append(acc_func(predictions, target).item())

        train_acc = mean(train_acc_batches)
        log_loss_mean = mean(log_loss_batches)
        
        # Update temperature
        #curr_temp = max(lambda_temp*curr_temp, min_temp)
        if curr_temp == min_temp:
            print("MINIMUM TEMPERATURE REACHED")
            
        test_acc = evaluate_accuracy_next_activity(rnn, test_dataset, acc_func)
        
        if epoch % 100 == 0:
            print("Epoch {}:\tlogic_loss:{}\ttrain accuracy:{}\ttest accuracy:{}".format(epoch, log_loss_mean, train_acc, test_acc))

        if log_loss_mean < epsilon:
            return train_acc, test_acc

        if epoch > 500 and abs(log_loss_mean - old_loss) < 0.00001:
            return train_acc, test_acc
        old_loss = log_loss_mean

    return train_acc, test_acc