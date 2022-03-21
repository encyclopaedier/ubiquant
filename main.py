from train_eval.train import *

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = Network(args['context']).to(device)
    optimizer = optim.Adam(model.parameters(), lr=args['lr'])
    criterion = torch.nn.CrossEntropyLoss()
    # If you want to use full Dataset, please pass None to csvpath
    train_samples = LibriSamples(data_path=args['LIBRI_PATH'], shuffle=True, partition="train-clean-100",
                                 csvpath=None)  # "/content/train_filenames_subset_0008_v2.csv"
    dev_samples = LibriSamples(data_path=args['LIBRI_PATH'], shuffle=True, partition="dev-clean")

    for epoch in range(1, args['epoch'] + 1):
        train(args, model, device, train_samples, optimizer, criterion, epoch)
        test_acc = test(args, model, device, dev_samples)
        print('Dev accuracy ', test_acc)
    prediction_samples = TestSamples(data_path=args['LIBRI_PATH'], partition="test-clean")
    predict(args, model, device, prediction_samples, test_acc)