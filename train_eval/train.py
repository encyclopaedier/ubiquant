from config.config import *
from Dataloader.dataloader import *
from tqdm import tqdm
import torch.optim as optim
class Train():
    def __init__(self,model,optimizer,criterion,scaler,batch_size,path,epoch,num_workers):
        self.model=model
        self.optimizer=optimizer
        self.criterion=criterion
        self.scaler=scaler
        self.batch_size=batch_size
        self.path=path
        self.epochs=epoch
        self.num_workers=num_workers
        self.init_device()
        self.init_librisample()

    def init_device(self):
        self.device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    def init_librisample(self):
        self.librisample = LibriSamples(self.path)

    def train(self):

        for epoch in range(self.epochs):
            total_loss = 0
            total_length = 1
            for i in range(len(self.librisample)):
                X, Y = self.librisample[i]
                train_items = LibriItems(X, Y)
                train_loader = torch.utils.data.DataLoader(train_items, batch_size=self.batch_size,
                                                           shuffle=True,num_workers=self.num_workers)  # num_workers=1
                self.scheduler = optim.lr_scheduler.CosineAnnealingLR(self.optimizer,
                                                                      T_max=(len(train_loader) * self.epochs))
                total_length+=len(train_loader)
                batch_bar = tqdm(total=len(train_loader), dynamic_ncols=True, leave=False, position=0, desc='Train')

                for batch_idx, (data, target) in enumerate(train_loader):
                    data = data.float().to(self.device)
                    target = target.long().to(self.device)

                    # Don't be surprised - we just wrap these two lines to make it work for FP16
                    with torch.cuda.amp.autocast():
                        outputs = self.model(data)
                        loss = self.criterion(outputs, target)

                    self.optimizer.step()
                    total_loss += float(loss)
                    batch_bar.set_postfix(
                        loss="{:.04f}".format(float(total_loss / (i + 1))),
                        lr="{:.04f}".format(float(self.optimizer.param_groups[0]['lr'])))
                    # Another couple things you need for FP16.
                    self.scaler.scale(loss).backward()  # This is a replacement for loss.backward()
                    self.scaler.step(self.optimizer)  # This is a replacement for optimizer.step()
                    self.scaler.update()  # This is something added just for FP16

                    self.scheduler.step()  # We told scheduler T_max that we'd call step() (len(train_loader) * epochs) many times.

                    batch_bar.update()  # Update tqdm bar

            # You can add validation per-epoch here if you would like

            print("Epoch {}/{}: Train Loss {:.04f}, Learning Rate {:.04f}".format(
                epoch + 1,
                self.epochs,
                float(total_loss / total_length),
                float(self.optimizer.param_groups[0]['lr'])))

if __name__=="__main__":
    train=Train()