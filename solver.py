import os
import torch
import time
import datetime


class Solver(object):

    DEFAULTS = {}

    def __init__(self, version, data_loader, config):
        """
        Initializes a Solver object
        """

        # data loader
        self.__dict__.update(Solver.DEFAULTS, **config)
        self.version = version
        self.data_loader = data_loader

        self.build_model()

        # TODO: build tensorboard

        # start with a pre-trained model
        if self.pretrained_model:
            self.load_pretrained_model()

    def build_model(self):
        """
        Instantiates the model, loss criterion, and optimizer
        """

        # TODO: instantiate model

        # TODO: instantiate loss criterion

        # TODO: instantiate optimizer

        # TODO: print network
        # self.print_network(self.model, '')

        # TODO: use gpu if enabled
        # if torch.cuda.is_available() and self.use_gpu:
        #    self.model.cuda()
        #    self.criterion.cuda()

    def print_network(self, model, name):
        """
        Prints the structure of the network and the total number of parameters
        """
        num_params = 0
        for p in model.parameters():
            num_params += p.numel()
        print(name)
        print(model)
        print("The number of parameters: {}".format(num_params))

    def load_pretrained_model(self):
        """
        loads a pre-trained model from a .pth file
        """
        self.model.load_state_dict(torch.load(os.path.join(
            self.model_save_path, '{}.pth'.format(self.pretrained_model))))
        print('loaded trained model ver {}'.format(self.pretrained_model))

    def print_loss_log(self, start_time, iters_per_epoch, e, i, loss):
        """
        Prints the loss and elapsed time for each epoch
        """
        total_iter = self.num_epochs * iters_per_epoch
        cur_iter = e * iters_per_epoch + i

        elapsed = time.time() - start_time
        total_time = (total_iter - cur_iter) * elapsed / (cur_iter + 1)
        epoch_time = (iters_per_epoch - i) * elapsed / (cur_iter + 1)

        epoch_time = str(datetime.timedelta(seconds=epoch_time))
        total_time = str(datetime.timedelta(seconds=total_time))
        elapsed = str(datetime.timedelta(seconds=elapsed))

        log = "Elapsed {}/{} -- {}, Epoch [{}/{}], Iter [{}/{}], " \
              "loss: {:.4f}".format(elapsed,
                                    epoch_time,
                                    total_time,
                                    e + 1,
                                    self.num_epochs,
                                    i + 1,
                                    iters_per_epoch,
                                    loss)

        # TODO: add tensorboard

        print(log)

    def save_model(self, e):
        """
        Saves a model per e epoch
        """
        path = os.path.join(
            self.model_save_path,
            '{}/{}.pth'.format(self.version, e + 1)
        )

        torch.save(self.model.state_dict(), path)

    def model_step(self, images, labels):
        """
        A step for each iteration
        """

        # TODO: set model in training mode
        # self.model.train()

        # TODO: empty the gradients of the model through the optimizer
        # TODO: self.optimizer.zero_grad()

        # TODO: forward pass
        # TODO: output = self.model(images)

        # TODO: compute loss
        # TODO: loss = self.criterion(output, labels.squeeze())

        # TODO: compute gradients using back propagation
        # loss.backward()

        # TODO: update parameters
        # self.optimizer.step()

        # TODO: return loss
        # return loss
        pass

    def train(self):
        """
        Training process
        """
        # TODO: add training process
        pass

    def eval(self, data_loader):
        """
        Returns the count of top 1 and top 5 predictions
        """

        # set the model to eval mode
        # TODO: self.model.eval()

        # TODO: return evaluation metric
        pass

    def train_evaluate(self, e):
        """
        Evaluates the performance of the model using the train dataset
        """
        # TODO: call self.eval() then print log
        pass

    def test(self):
        """
        Evaluates the performance of the model using the test dataset
        """
        # TODO: call self.eval() then print log
        pass
