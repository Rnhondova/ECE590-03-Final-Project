from __future__ import print_function, division

import torch, gc
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
import matplotlib.pyplot as plt
import time
import os
import copy
from PIL import Image
from torch.utils.tensorboard import SummaryWriter

from tools.data_loader import *
from tools.initialize_model import *

def initialize_dataloader(input_size, data_dir, image_index_file_location):
  data_transforms = {
      'train': transforms.Compose([
          transforms.RandomResizedCrop(input_size),
          transforms.RandomHorizontalFlip(),
          transforms.ToTensor(),
          transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
      ]),
      'val': transforms.Compose([
          transforms.Resize(input_size),
          transforms.CenterCrop(input_size),
          transforms.ToTensor(),
          transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
      ]),
  }

  image_datasets = {x: LoadDataset(text_file = image_index_file_location, root_dir = data_dir,
                                  transform=data_transforms[x],train_or_val=x)
                    for x in ['train', 'val']}

  dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=64,
                                              shuffle=True, num_workers=4)
                 for x in ['train', 'val']}

  dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val']}
  class_names = image_datasets['train'].classes

  return dataloaders

def train_model(model, criterion, optimizer, scheduler, device, dataloaders = '',num_epochs=25,model_save_name='model_ft.h5',
                chk_point_path ='',writer = '',dataset_sizes=0, return_best_results_only= True,start_epoch = 0):
    since = time.time()

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0
    n_total_steps = len(dataloaders['train'])

    loss_cml_tr = {}
    accuracy_cml_tr = {}
    loss_cml_val = {}
    accuracy_cml_val = {}


    for epoch in range(start_epoch, num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # Set model to training mode

            else:
                model.eval()   # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0

            # Iterate over data.
            for result_ in dataloaders[phase]:
                
                
                inputs, labels = result_['image'],result_['labels']
                inputs = inputs.to(device)
                labels = labels.type(torch.LongTensor).to(device)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)
                

               

            if phase == 'train':
                scheduler.step()

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]

            print('{} Loss: {:.4f} Acc: {:.4f}'.format(
                phase, epoch_loss, epoch_acc))
            
            if phase == 'train':
                writer.add_scalar('training loss', epoch_loss,epoch)
                writer.add_scalar('training accuracy', epoch_acc,epoch)
                loss_cml_tr.update({epoch : epoch_loss})
                accuracy_cml_tr.update({epoch : epoch_acc})
                
            else:
                writer.add_scalar('validation loss', epoch_loss,epoch)
                writer.add_scalar('validation accuracy', epoch_acc,epoch)
                loss_cml_val.update({epoch : epoch_loss})
                accuracy_cml_val.update({epoch : epoch_acc})

            # deep copy the model
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())

                # Save for checkpoint
                if not os.path.exists(chk_point_path):
                    os.makedirs(chk_point_path)
                print("Saving ...")
                state = {'net': model.state_dict(),
                        'epoch': epoch,
                        'lr': scheduler.get_last_lr()}
                torch.save(state, os.path.join(chk_point_path, model_save_name))
         
        print()

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))

    if return_best_results_only:
      model = None
      writer.flush()

      # Save
      save_dir_pnts = '{}/saved_training_points'.format(chk_point_path)
      os.makedirs(save_dir_pnts, exist_ok=True)
      combined_stats = {'train loss': loss_cml_tr,
                        'train accuracy': accuracy_cml_tr,
                        'validation loss': loss_cml_val,
                        'validation accuracy': accuracy_cml_val}
      np.save('{}/{}.npy'.format(save_dir_pnts, model_save_name.replace(".h5", "")), combined_stats) 

      return best_acc

    # load best model weights
    model.load_state_dict(best_model_wts)
    
    return model

def imshow(inp, title=None):
    """Imshow for Tensor."""
    inp = inp.numpy().transpose((1, 2, 0))
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    inp = std * inp + mean
    inp = np.clip(inp, 0, 1)
    plt.imshow(inp)
    if title is not None:
        plt.title(title)
    plt.pause(0.001)  # pause a bit so that plots are updated

def visualize_model(model, device, dataloaders,class_names,num_images=6, return_only_images = True):
    was_training = model.training
    model.eval()
    images_so_far = 0
    fig = plt.figure()
    predictions = []
    actual_class = []
    batch_num = 0
    with torch.no_grad():
        for result_ in dataloaders['val']:
          batch_num += 1
          print('Starting batch: %s' % (batch_num))
          inputs, labels = result_['image'],result_['labels']
          inputs = inputs.to(device)
          labels = labels.to(device)

          outputs = model(inputs)
          _, preds = torch.max(outputs, 1)

          if batch_num > 1:
            predictions = np.array([*predictions.copy(),*preds.cpu().numpy().copy()])
            actual_class = np.array([*actual_class.copy(),*labels.cpu().numpy().copy()])
          else:
            predictions = preds.cpu().numpy().copy()
            actual_class = labels.cpu().numpy().copy()

          if images_so_far < num_images:
            for j in range(inputs.size()[0]):
                images_so_far += 1
                ax = plt.subplot(num_images//2, 2, images_so_far)
                ax.axis('off')
                ax.set_title('predicted: {}'.format(class_names[preds[j]]))
                imshow(inputs.cpu().data[j])

                #writer.add_image('car_damage_images', out)

                if images_so_far == num_images and return_only_images == True:
                  model.train(mode=was_training)
                  return
                elif images_so_far == num_images:
                  break
        model.train(mode=was_training)
    return actual_class, predictions

def set_up_training_schedule(inputs, 
                             model_name = 'resnet50', 
                             number_of_classes= 12,
                             device='', 
                             INITIAL_LR=0.01, 
                             writer='', 
                             step_after =10,
                             transfer_learning_type = 'Fixed Features',
                             proportion_fixed = 0,
                             number_of_pochs=35, 
                             model_save_name = 'model_conv.h5',
                             CHECKPOINT_PATH="runs",
                             optimizer_to_use = "SGD",
                             dataset_sizes = 0,
                             return_best_results_only = False,
                             train_from_scratch = True,
                             data_dir = "",
                             image_index_file_location = ""):

  # Initialize the model for this run
  model_ft, input_size = initialize_model(model_name = model_name, num_classes = number_of_classes, feature_extract=transfer_learning_type, use_pretrained=True, proportion_fixed=proportion_fixed)

  dataloaders = initialize_dataloader(input_size, data_dir, image_index_file_location)

  start_epoch = 0
  if not train_from_scratch:
    model_ft, start_epoch, INITIAL_LR = load_training_starting_point(model_ft,os.path.join(CHECKPOINT_PATH, model_save_name),train_from_scratch = train_from_scratch,INITIAL_LR = INITIAL_LR)

  model_ft = model_ft.to(device)

  criterion = nn.CrossEntropyLoss()

  # Observe that all parameters are being optimized
  if optimizer_to_use == "SGD":
    optimizer = optim.SGD(model_ft.parameters(), lr=INITIAL_LR, momentum=0.9)
  else:
    optimizer = optim.Adam(model_ft.parameters(), lr=INITIAL_LR)

  # Decay LR by a factor of 0.1 every 7 epochs
  exp_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=step_after, gamma=0.1)

  #writer.add_graph(model_ft,inputs.to(device))
  #writer.flush()
  #writer.close()
  
  model_ft = train_model(model_ft, criterion, optimizer,
                         exp_lr_scheduler, device, num_epochs=number_of_pochs,model_save_name=model_save_name,
                         chk_point_path =CHECKPOINT_PATH, dataloaders=dataloaders, writer = writer,
                         dataset_sizes = dataset_sizes, return_best_results_only=return_best_results_only,
                         start_epoch = start_epoch)

  return model_ft

def get_checkpoint(ckpt_path):
    try:
        ckpt = torch.load(ckpt_path)
    except Exception as e:
        print(e)
        return None
    return ckpt

def load_training_starting_point(model,CKPT_PATH,train_from_scratch = True,INITIAL_LR = 0.01):
  ckpt = get_checkpoint(CKPT_PATH)
  if ckpt is None or train_from_scratch:
      if not train_from_scratch:
          print("Checkpoint not found.")
      print("Training from scratch ...")
      start_epoch = 0
      current_learning_rate = INITIAL_LR
  else:
      print("Successfully loaded checkpoint: %s" %CKPT_PATH)
      model.load_state_dict(ckpt['net'])
      start_epoch = ckpt['epoch'] + 1
      current_learning_rate = ckpt['lr'][0]
      print("Starting from epoch %d " %start_epoch)

  print("Starting from learning rate %f:" %current_learning_rate)
  return model, start_epoch, current_learning_rate

def train_multiple_variants(inputs, device, writer_path,
                            pretrained_model_name=['resnet50'],
                            variant=["finetune"], 
                            number_of_classes= 12,
                            INITIAL_LR=[0.01],
                            number_of_pochs=35, 
                            CHECKPOINT_PATH="runs",
                            step_after = 10,
                            proportion_fixed = 0.5,
                            optimization=['SGD'],
                            dataset_sizes = 0,
                            train_from_scratch = True,
                            data_dir = "",
                            image_index_file_location = ""
                            ):
  
  try:
    model_results = {}
    for model_type in pretrained_model_name:
      for transfer_ln_type in variant:
        for optim_type in optimization:
          for learning_rate_val in INITIAL_LR:
            lr_str = str(learning_rate_val)
            lr_str = lr_str.replace(".", "pt")

            model_name = '{}_{}_{}_{}.h5'.format(model_type,transfer_ln_type,optim_type,lr_str)
            writer = SummaryWriter('{}/{}'.format(writer_path,model_name))

            print("Starting to train model: {}".format(model_name))

            model_val_acc = set_up_training_schedule(inputs=inputs,
                                                model_name = model_type, 
                                                number_of_classes= number_of_classes,
                                                device=device, 
                                                INITIAL_LR=learning_rate_val, 
                                                writer=writer, 
                                                step_after = step_after,
                                                transfer_learning_type = transfer_ln_type,
                                                proportion_fixed = proportion_fixed,
                                                number_of_pochs=number_of_pochs, 
                                                model_save_name = model_name,
                                                CHECKPOINT_PATH=CHECKPOINT_PATH,
                                                optimizer_to_use = optim_type,
                                                dataset_sizes = dataset_sizes,
                                                return_best_results_only = True,
                                                train_from_scratch = train_from_scratch,
                                                data_dir = data_dir,
                                                image_index_file_location = image_index_file_location)
            model_results.update({model_name : model_val_acc})
          
            print("Finished training model: {}".format(model_name))
      torch.cuda.empty_cache()
      gc.collect()
  except Exception as e:
    print(e)
    return model_results
  
  return model_results


