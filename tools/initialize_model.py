from __future__ import print_function
import torch
import torch.nn as nn
import torchvision
from torchvision import models
import numpy as np

def set_gradients(model,transfer_learning_type,proportion_fixed=0.5):
    if transfer_learning_type == 'fixedfeatures':
      for param in model.parameters():
        param.requires_grad = False
    elif transfer_learning_type == 'hybridfixedfeatures':
      assert(proportion_fixed>=0 and proportion_fixed<=1)
      
      # Counting number of layers
      num_layers = 0
      for name, param in model.named_parameters(): 
          num_layers += 1
      
      # Fixed required proportion
      number_of_layers_fixed = np.round(num_layers*proportion_fixed)
      layers_fixed = 0

      for param in model.parameters():
        if layers_fixed <= number_of_layers_fixed:
          param.requires_grad = False
          layers_fixed +=1

def initialize_model(model_name, num_classes, feature_extract, use_pretrained=True,proportion_fixed=0.5):
    # Initialize these variables which will be set in this if statement. Each of these
    #   variables is model specific.
    model_ft = None
    input_size = 0

    if model_name == "resnet18":
        """ Resnet18
        """
        model_ft = models.resnet18(pretrained=use_pretrained)
        set_gradients(model_ft, feature_extract, proportion_fixed = proportion_fixed)
        num_ftrs = model_ft.fc.in_features
        model_ft.fc = nn.Linear(num_ftrs, num_classes)
        input_size = 224
    
    elif model_name == "resnet50":
        """ Resnet50
        """
        model_ft = models.resnet50(pretrained=use_pretrained)
        set_gradients(model_ft, feature_extract, proportion_fixed = proportion_fixed)
        num_ftrs = model_ft.fc.in_features
        model_ft.fc = nn.Linear(num_ftrs, num_classes)
        input_size = 224

    elif model_name == "resnext101_32x8d":
        """ Resnext101 32x8d
        """
        model_ft = models.resnext101_32x8d(pretrained=use_pretrained)
        set_gradients(model_ft, feature_extract, proportion_fixed = proportion_fixed)
        num_ftrs = model_ft.fc.in_features
        model_ft.fc = nn.Linear(num_ftrs, num_classes)
        input_size = 224

    elif model_name == "wide_resnet50_2":
        """ Wide Resnet50
        """
        model_ft = models.wide_resnet50_2(pretrained=use_pretrained)
        set_gradients(model_ft, feature_extract, proportion_fixed = proportion_fixed)
        num_ftrs = model_ft.fc.in_features
        model_ft.fc = nn.Linear(num_ftrs, num_classes)
        input_size = 224


    elif model_name == "alexnet":
        """ Alexnet
        """
        model_ft = models.alexnet(pretrained=use_pretrained)
        set_gradients(model_ft, feature_extract, proportion_fixed = proportion_fixed)
        num_ftrs = model_ft.classifier[6].in_features
        model_ft.classifier[6] = nn.Linear(num_ftrs,num_classes)
        input_size = 224

    elif model_name == "vgg11_bn":
        """ VGG11_bn
        """
        model_ft = models.vgg11_bn(pretrained=use_pretrained)
        set_gradients(model_ft, feature_extract, proportion_fixed = proportion_fixed)
        num_ftrs = model_ft.classifier[6].in_features
        model_ft.classifier[6] = nn.Linear(num_ftrs,num_classes)
        input_size = 224

    elif model_name == "vgg19_bn":
        """ VGG19_bn
        """
        model_ft = models.vgg19_bn(pretrained=use_pretrained)
        set_gradients(model_ft, feature_extract, proportion_fixed = proportion_fixed)
        num_ftrs = model_ft.classifier[6].in_features
        model_ft.classifier[6] = nn.Linear(num_ftrs,num_classes)
        input_size = 224

    elif model_name == "squeezenet1_0":
        """ Squeezenet
        """
        model_ft = models.squeezenet1_0(pretrained=use_pretrained)
        set_gradients(model_ft, feature_extract, proportion_fixed = proportion_fixed)
        model_ft.classifier[1] = nn.Conv2d(512, num_classes, kernel_size=(1,1), stride=(1,1))
        model_ft.num_classes = num_classes
        input_size = 224

    elif model_name == "densenet121":
        """ Densenet
        """
        model_ft = models.densenet121(pretrained=use_pretrained)
        set_gradients(model_ft, feature_extract, proportion_fixed=proportion_fixed)
        num_ftrs = model_ft.classifier.in_features
        model_ft.classifier = nn.Linear(num_ftrs, num_classes)
        input_size = 224

    elif model_name == "inception_v3":
        """ Inception v3
        Be careful, expects (299,299) sized images and has auxiliary output
        """
        model_ft = models.inception_v3(pretrained=use_pretrained)
        set_gradients(model_ft, feature_extract, proportion_fixed = proportion_fixed)
        # Handle the auxilary net
        num_ftrs = model_ft.AuxLogits.fc.in_features
        model_ft.AuxLogits.fc = nn.Linear(num_ftrs, num_classes)
        # Handle the primary net
        num_ftrs = model_ft.fc.in_features
        model_ft.fc = nn.Linear(num_ftrs,num_classes)
        input_size = 299

    else:
        print("Invalid model name, exiting...")
        exit()

    return model_ft, input_size
