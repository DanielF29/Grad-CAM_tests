#!/usr/bin/env python
# coding: utf-8
#
# Author:   Kazuto Nakashima
# URL:      http://kazuto1011.github.io
# Created:  2017-05-18

from __future__ import print_function

import copy
import os.path as osp

import click
import cv2
import matplotlib.cm as cm
import numpy as np
import torch 
import torch.nn.functional as F
from torchvision import models, transforms

from grad_cam import (
    BackPropagation,
    Deconvnet,
    GradCAM,
    GuidedBackPropagation,
    occlusion_sensitivity,
)

import glob #to ...obtain a list of certain types of files at a dicrectory..right?
import os 

# if a model includes LSTM, such as in image captioning,
# torch.backends.cudnn.enabled = False

def get_device(cuda):
    #print("cuda is equal to:",cuda)
    #CudaAvailability = torch.cuda.is_available()
    #print("cuda.is_available is equal to:", CudaAvailability)
    #cuda = True   #se agrego para forzar que se use la GPU
    #cuda = cuda and torch.cuda.is_available() #codigo original
    #cuda = torch.cuda.is_available()          #codigo modificado para seleccionar la GPU siempre que este disponible
    device = torch.device("cuda" if cuda else "cpu") #codigo Original
    #device = torch.device("cuda")  #codigo modificado para forzar que se use la GPU
    if cuda:
        current_device = torch.cuda.current_device()
        print(" Device:", torch.cuda.get_device_name(current_device))
    else:
        #print("Device: CPU")  #codigo original
        print(" Device:", torch.cuda.get_device_name(current_device)) #para forzar que imprima el divice usado
    return device

def load_images(image_paths):
    images = []
    raw_images = []
    #print("Images:")
    for i, image_path in enumerate(image_paths):
        #print("\t#{}: {}".format(i, image_path))
        image, raw_image = preprocess(image_path)
        images.append(image)
        raw_images.append(raw_image)
    return images, raw_images

def preprocess(image_path):
    #try:
    #    path=os.path.join(image_path,n)
    #    raw_image=cv2.imread(path)
    #    raw_image=cv2.resize(raw_image, (224,224))

    #except Exception as e:
    #    print(str(e))
    #print("-->>>> image_path:{} <<<<------".format(image_path))
    raw_image = cv2.imread(image_path)
    raw_image = cv2.resize(raw_image, (224,) * 2)
    image = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.614, 0.481, 0.3006], std=[0.0812, 0.0787, 0.0751]),
            #transforms.Normalize(mean=[0.485, 0.456, 0.406],  std=[0.229, 0.224, 0.225]),
        ]
    )(raw_image[..., ::-1].copy())
    return image, raw_image

def save_gradient(filename, gradient):
    gradient = gradient.cpu().numpy().transpose(1, 2, 0)
    gradient -= gradient.min()
    if not gradient.max() == 0:
        gradient /= gradient.max()
    #cmap = cm.jet_r(gradient)[..., :3]* 255.0
    gradient *= 255.0

    cv2.imwrite(filename, np.uint8(gradient))

def save_gradcam(filename, gcam, raw_image, paper_cmap=False, valance=0.5):
    #print("  ")
    #print("Save_GradCAM input:  ")
    #print("Shape of gcam: {}, Max value of gcam: {},  Min value of gcam: {}". format(
         ##gcam.shape, np.max(gcam), np.min(gcam)
    #    gcam.size(), torch.max(gcam), torch.min(gcam)
    #    )
    #)
    gcam = gcam.cpu().numpy()
    #print("gcam = gcam.cpu().numpy() and then:")
    #print("Shape of gcam: {}, Max value of gcam: {},  Min value of gcam: {}". format(
    #    gcam.shape, np.max(gcam), np.min(gcam)
    #    )
    #)
    #print("  ")
    cmap = cm.jet_r(gcam)[..., :3] * 255.0
    if paper_cmap:
        alpha = gcam[..., None]
        alpha = alpha * valance
        gcam = alpha * cmap + (1 - alpha) * raw_image

    else:
        #gcam = (cmap.astype(np.float) + raw_image.astype(np.float)) / 2
        gcam = (     valance*(cmap.astype(np.float)) + (1-valance)*(raw_image.astype(np.float))   )
    cv2.imwrite(filename, np.uint8(gcam))

def save_sensitivity(filename, maps):
    maps = maps.cpu().numpy()
    scale = max(maps[maps > 0].max(), -maps[maps <= 0].min())
    maps = maps / scale * 0.5
    maps += 0.5
    maps = cm.bwr_r(maps)[..., :3]
    maps = np.uint8(maps * 255.0)
    maps = cv2.resize(maps, (224, 224), interpolation=cv2.INTER_NEAREST)
    cv2.imwrite(filename, maps)

def gpu_space(prevFree_GPUram = 0.0, all = False):
    MegaBites= 1024*1024
    t = torch.cuda.get_device_properties(0).total_memory
    t = t/MegaBites
    if all:
        print(" --> Total GPU-VRAM en GB:{}".format( t/1024 ))

    r = torch.cuda.memory_reserved(0) 
    r = (r/MegaBites)
    a = torch.cuda.memory_allocated(0)
    a = (a/MegaBites)
    f = r-a  # free inside reserved ...in MB
    if not prevFree_GPUram == 0.0:
        print(" Reduction on Free_GPUram: {:.3f}MB".format( (prevFree_GPUram - f) ))
    #print("------<<<<<------<<<<<-----XXX----->>>>>-------->>>>>------")
    print(" Free: Reserved:{:.3f}-Allocated:{:.3f}= {:.3f}MB".format(r, a, f ))
    print(" Total({}GB)-Reserved=> En {:.3f}MB, En {:.3f}GB".format(
        t/1024,                    (t-r),     (t-r)/1024
        )
    )
    #print("    ")
    return f

def loading_NNModel(namemodel_loaded, num_classes=4 ):
    #Instruction to read a model from a "xxx.ckpt" file 
    #model = AlexnetModel(hparams={"lr": 0.00005}, num_classes=4, pretrained=False, seed=None) #seed=manualSeed)   #<<<<<<<<<<<<<<<<-----<<<<----<<<---
    #model = Vgg16Model(hparams={"lr": 0.00005}, num_classes=4, pretrained=False, seed=None)
    #model = Vgg16Model(hparams={}, num_classes=4, pretrained=False, seed=None)
    model = models.vgg16()
    model.classifier = torch.nn.Sequential(
        torch.nn.Linear(25088, 4096),  torch.nn.ReLU(),
        torch.nn.Dropout(p=0.5),       torch.nn.Linear(4096, 256),
        torch.nn.ReLU(),               torch.nn.Dropout(p=0.5),
        torch.nn.Linear(256, num_classes)
        )
    model_loaded = torch.load("{}".format(namemodel_loaded))
    print ("            >>>>>>>>> model <<<<<<<<<<<<")
    print(model)
    for l in model_loaded:
        #print(l)
        if l == "state_dict":
            corrected_dict = {}
            for layerName in model_loaded["state_dict"]:
                NewLayerName = layerName.split(".", 1)[1]
                #print(NewLayerName)
                corrected_dict[NewLayerName] = model_loaded["state_dict"][layerName]
            print (" >>>>>>>>> Loading the state_dict of {} <<<<<<<<<<<<".format(namemodel_loaded))
            #print(corrected_dict)
            model.load_state_dict(corrected_dict)
    #del model_loaded
    #del corrected_dict
    #del layerName
    #del NewLayerName
    #del NameModelLoaded

    return model

def printing_spaces():
    print(" ------------------------------------")
    print(" ------------------------------------")
    print(" ")
    print(" ")

def Imags_dir_and_name(filename):
    Image_Name = filename.split("\\")[-1]
    Image_Name = Image_Name.split("/")[-1]
    Image_Name = Image_Name.split(".",1)[0]
    #print("File_Name: {}".format(filename) )
    #print("CutImgName: {}, CutImgType:{}".format(Image_Name, type(Image_Name))  )
    print(" Image_Name: {}".format(Image_Name))
    dir = [filename]

    return Image_Name, dir


@click.group()
@click.pass_context
def main(ctx):
    print(" Mode:", ctx.invoked_subcommand)
    #torch.cuda.empty_cache()

@main.command()
#@click.option("-i", "--images_folder_path", type=str, multiple=True, required=True)
@click.option("-i", "--images_folder_path", type=str, required=True)
@click.option("-t", "--namemodel_loaded", type=str, required=True)
#@click.option("-c", "--claseToEval", type=int, required=True)
@click.option("-o", "--output-dir", type=str, default="./results")
@click.option("--cuda/--cpu", default=True)
#def demo0(image_paths, claseToEval, output_dir, cuda):
def demo0(images_folder_path, namemodel_loaded, output_dir, cuda):
    device = get_device(cuda) #check if their is a available GPU
    #prints the total GPU RAM, 
    #reserved, availableallocated, free and reduced.
    prevFree_GPUram = gpu_space(all = True, prevFree_GPUram = 0.0)                
    num_classes=4
    targeted_layers = ["avgpool"]

    # Model
    #  Names of the NN models to be loaded
    #namemodel_loaded = "vgg16_4c_combined.ckpt"
    #namemodel_loaded = "vgg16_4c_sections.ckpt"
    #namemodel_loaded = "vgg16_4c_surface.ckpt"
    model = loading_NNModel(namemodel_loaded, num_classes)
    model.to(device)
    model.eval()

    prevFree_GPUram = gpu_space(prevFree_GPUram)
    print(" ------<<<<<------<<<<<-----Before FORs----->>>>>-------->>>>>------")
    for filename in glob.glob('{}/*.png'.format(images_folder_path)):
        printing_spaces()
        #  The "filename"(path to an image) is given to obtain: 
        #the name of the image and 
        #the path to the image is returned as a list
        Image_Name, dir = Imags_dir_and_name(filename)

        #  Images  
        images, raw_images = load_images(dir)
        images = torch.stack(images).to(device)

        #  Grad-CAM
        gcam = GradCAM(model=model)#, target_layer= targeted_layers) #Suponiendo que se retendran menos "hooks"
        probs, ids = gcam.forward(images)

        # Guided Backpropagation
        gbp = GuidedBackPropagation(model=model)
        _ = gbp.forward(images)

        for claseToEval in range(4):
            gbp.backward(ids=ids[:, [claseToEval]])
            gcam.backward(ids=ids[:, [claseToEval]])
            
            for target_layer in targeted_layers:
                #ids_ = torch.LongTensor([[claseToEval]] * len(images)).to(device)    #target_class]] * len(images)).to(device)
                #gcam.backward(ids=ids_)
                gradients = gbp.generate()
                regions = gcam.generate(target_layer= target_layer)
                """
                print("Type of regions: {}, shape: {}, Max value of regions: {},  Min value of regions: {}". format(
                    type(regions),regions.size(), torch.max(regions), torch.min(regions) 
                    )
                ) 
                #"""

                save_gradcam(
                    filename=osp.join(
                        output_dir,
                        #"{}_{}_Class-{}_({:.5f}).png".format(
                        "VGG16-avgpool-{}-C{}({:.5f}).png".format(
                            Image_Name, claseToEval, float(probs[ ids == claseToEval])
                        ),
                    ),
                    #gcam=regions[j, 0],
                    gcam=regions[0, 0],
                    #raw_image=raw_images[j],
                    raw_image=raw_images[0],
                    paper_cmap=True, valance=0.3
                )
                # Guided Grad-CAM
                save_gradient(
                    filename=osp.join(
                        output_dir,
                        #"{}_{}_Class-{}_({:.5f}).png".format(
                        "VGG16-avgpool-{}-C{}({:.5f})_Guided.png".format(
                            Image_Name, claseToEval, float(probs[ids == claseToEval])
                        ),
                    ),
                    gradient=torch.mul(regions, gradients)[0]
                    #gradient=torch.mul(regions, gradients)[0],
                    #raw_image=raw_images[0],
                    #paper_cmap=True, valance=0.3
                )
                """
                # Guided Grad-CAM
                save_gradient(
                    filename=osp.join(
                        output_dir,
                        "Guided_GCAM_VGG16-avgpool-{}-C{}({:.5f}).png".format(
                            Image_Name, claseToEval, float(probs[ids == claseToEval])
                        ),
                    ),
                    gradient=torch.mul(regions, gradients)[0],
                )
                #"""
        prevFree_GPUram = gpu_space(prevFree_GPUram)
        """
        #Borrado de variables del ultimo ciclo "For"
        gcam.remove_hook()
        del gcam
        del images
        del raw_images
        del probs
        del ids
        del ids_
        del regions
        # """
    ##############################
    #End of 3 FORs   #############
    ##############################
    del model
    #with torch.no_grad():
    #    torch.cuda.empty_cache()
    torch.cuda.empty_cache()
    print("          -------->>><<<<-----------")
    print("----------      Fin de Demo0        ")
    print("          -------->>><<<<-----------")


@main.command()
#@click.option("-i", "--images_folder_path", type=str, multiple=True, required=True)
@click.option("-i", "--images_folder_path", type=str, required=True)
@click.option("-t", "--namemodel_loaded", type=str, required=True)
@click.option("-k", "--clase_to_eval", type=int, required=True)
@click.option("-o", "--output-dir", type=str, default="./results")
@click.option("--cuda/--cpu", default=True)
#def demo0(image_paths, claseToEval, output_dir, cuda):
def democlass(images_folder_path, namemodel_loaded, clase_to_eval, output_dir, cuda):
    print("Demo Class running")
    gpu_space(all = True) #prints the total GPU RAM, 
                          #reserved, availableallocated, free and reduced.

    #  Now we will set some variables:
    device = get_device(cuda)
    num_classes=4 #Output clases of the NN model.
    targeted_layers = ["avgpool"] #Layers to generated GradCAM on.
    claseToEval = clase_to_eval #3 #clase to be evaluated out of the 4 possible.

    #  Names of the NN models to be loaded
    #NameModelLoaded = "vgg16_4c_combined.ckpt"
    #NameModelLoaded = "vgg16_4c_sections.ckpt"
    #NameModelLoaded = "vgg16_4c_surface.ckpt"

    #  Model  
    model = loading_NNModel(namemodel_loaded, num_classes)
    model.to(device)
    model.eval()

    gpu_space()
    print("------<<<<<------<<<<<-----Before FORs----->>>>>-------->>>>>------")
    for filename in glob.glob('{}/*.png'.format(images_folder_path)):
        printing_spaces()

        Image_Name = filename.split("\\")[-1]
        Image_Name = Image_Name.split("/")[-1]
        Image_Name = Image_Name.split(".",1)[0]
        #print("File_Name: {}".format(filename) )
        #print("CutImgName: {}, CutImgType:{}".format(Image_Name, type(Image_Name))  )
        print("Image_Name: {}".format(Image_Name))
        #print(" ")

        # Images  
        dir = [filename]
        images, raw_images = load_images(dir)
        images = torch.stack(images).to(device)
        #  Grad-CAM
        gcam = GradCAM(model=model)#, target_layer= targeted_layers) #Suponiendo que se retendran menos "hooks"
        probs, ids = gcam.forward(images)
        """
        print("Type of probs: {}, shape: {}, positions: {}, {}, {}, {}". format(
                type(probs),probs.size(), probs[0,0], probs[0,1], probs[0,2], probs[0,3]
            )
        ) 
        print("Probs tensor: {}".format(probs) )
        #"""
        gcam.backward(ids=ids[:, [claseToEval]])
        for target_layer in targeted_layers:
            #ids_ = torch.LongTensor([[claseToEval]] * len(images)).to(device)    #target_class]] * len(images)).to(device)
            #gcam.backward(ids=ids_)
            regions = gcam.generate(target_layer= target_layer)  
            #print("Type of regions: {}, shape: {}, Max value of regions: {},  Min value of regions: {}". format(
            #    type(regions),regions.size(), torch.max(regions), torch.min(regions) 
            #    )
            #)
            ######################

            save_gradcam(
                filename=osp.join(
                    output_dir,
                    "VGG16-avgpool--{}--C{}({:.5f}).png".format(
                        Image_Name, claseToEval, float(probs[ids == claseToEval])
                    ),
                ),
                #gcam=regions[j, 0],
                gcam=regions[0, 0],
                #raw_image=raw_images[j],
                raw_image=raw_images[0],
                paper_cmap=True, valance=0.3
            )
        gpu_space()
        """
        #  Borrado de variables del ultimo ciclo "For"
        gcam.remove_hook()
        del gcam
        del images
        del raw_images
        del probs
        del ids
        del ids_
        del regions
        # """
    ##############################
    #End of 3 FORs   #############
    ##############################
    del model
    torch.cuda.empty_cache()
    print("          -------->>><<<<-----------")
    print("----------      Fin de Demo Class3        ")
    print("          -------->>><<<<-----------")

@main.command()
#@click.option("-i", "--images_folder_path", type=str, multiple=True, required=True)
@click.option("-i", "--images_folder_path", type=str, required=True)
@click.option("-t", "--namemodel_loaded", type=str, required=True)
#@click.option("-c", "--claseToEval", type=int, required=True)
@click.option("-o", "--output-dir", type=str, default="./results")
@click.option("--cuda/--cpu", default=True)
#def demo0(image_paths, claseToEval, output_dir, cuda):
def democsv(images_folder_path, namemodel_loaded, output_dir, cuda):
    device = get_device(cuda) #check if their is a available GPU
    #prints the total GPU RAM, 
    #reserved, availableallocated, free and reduced.
    prevFree_GPUram = gpu_space(all = True, prevFree_GPUram = 0.0)                
    num_classes=4
    targeted_layers = ["avgpool"]

    # Model
    #  Names of the NN models to be loaded
    #namemodel_loaded = "vgg16_4c_combined.ckpt"
    #namemodel_loaded = "vgg16_4c_sections.ckpt"
    #namemodel_loaded = "vgg16_4c_surface.ckpt"
    model = loading_NNModel(namemodel_loaded, num_classes)
    model.to(device)
    model.eval()

    prevFree_GPUram = gpu_space(prevFree_GPUram)
    print(" ------<<<<<------<<<<<-----Before FORs----->>>>>-------->>>>>------")
    for filename in glob.glob('{}/*.png'.format(images_folder_path)):
        printing_spaces()
        #  The "filename"(path to an image) is given to obtain: 
        #the name of the image and 
        #the path to the image is returned as a list
        Image_Name, dir = Imags_dir_and_name(filename)

        #  Images  
        images, raw_images = load_images(dir)
        images = torch.stack(images).to(device)

        #  Grad-CAM
        gcam = GradCAM(model=model)#, target_layer= targeted_layers) #Suponiendo que se retendran menos "hooks"
        probs, ids = gcam.forward(images)

        # Guided Backpropagation
        gbp = GuidedBackPropagation(model=model)
        _ = gbp.forward(images)

        for claseToEval in range(4):
            gbp.backward(ids=ids[:, [claseToEval]])
            gcam.backward(ids=ids[:, [claseToEval]])
            
            for target_layer in targeted_layers:
                #ids_ = torch.LongTensor([[claseToEval]] * len(images)).to(device)    #target_class]] * len(images)).to(device)
                #gcam.backward(ids=ids_)
                gradients = gbp.generate()
                regions = gcam.generate(target_layer= target_layer)
                """
                print("Type of regions: {}, shape: {}, Max value of regions: {},  Min value of regions: {}". format(
                    type(regions),regions.size(), torch.max(regions), torch.min(regions) 
                    )
                ) 
                #"""

                save_gradcam(
                    filename=osp.join(
                        output_dir,
                        #"{}_{}_Class-{}_({:.5f}).png".format(
                        "VGG16-avgpool-{}-C{}({:.5f}).png".format(
                            Image_Name, claseToEval, float(probs[ ids == claseToEval])
                        ),
                    ),
                    #gcam=regions[j, 0],
                    gcam=regions[0, 0],
                    #raw_image=raw_images[j],
                    raw_image=raw_images[0],
                    paper_cmap=True, valance=0.3
                )
                # Guided Grad-CAM
                save_gradient(
                    filename=osp.join(
                        output_dir,
                        #"{}_{}_Class-{}_({:.5f}).png".format(
                        "VGG16-avgpool-{}-C{}({:.5f})_Guided.png".format(
                            Image_Name, claseToEval, float(probs[ids == claseToEval])
                        ),
                    ),
                    gradient=torch.mul(regions, gradients)[0]
                    #gradient=torch.mul(regions, gradients)[0],
                    #raw_image=raw_images[0],
                    #paper_cmap=True, valance=0.3
                )
                """
                # Guided Grad-CAM
                save_gradient(
                    filename=osp.join(
                        output_dir,
                        "Guided_GCAM_VGG16-avgpool-{}-C{}({:.5f}).png".format(
                            Image_Name, claseToEval, float(probs[ids == claseToEval])
                        ),
                    ),
                    gradient=torch.mul(regions, gradients)[0],
                )
                #"""
        prevFree_GPUram = gpu_space(prevFree_GPUram)
        """
        #Borrado de variables del ultimo ciclo "For"
        gcam.remove_hook()
        del gcam
        del images
        del raw_images
        del probs
        del ids
        del ids_
        del regions
        # """
    ##############################
    #End of 3 FORs   #############
    ##############################
    del model
    #with torch.no_grad():
    #    torch.cuda.empty_cache()
    torch.cuda.empty_cache()
    print("          -------->>><<<<-----------")
    print("----------      Fin de Demo CSV        ")
    print("          -------->>><<<<-----------")


if __name__ == "__main__":
    #torch.cuda.empty_cache()
    main()
    #torch.cuda.empty_cache()
