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
        print("Device:", torch.cuda.get_device_name(current_device))
    else:
        #print("Device: CPU")  #codigo original
        print("Device:", torch.cuda.get_device_name(current_device)) #para forzar que imprima el divice usado
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
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )(raw_image[..., ::-1].copy())
    return image, raw_image

def save_gradient(filename, gradient):
    gradient = gradient.cpu().numpy().transpose(1, 2, 0)
    gradient -= gradient.min()
    gradient /= gradient.max()
    gradient *= 255.0
    cv2.imwrite(filename, np.uint8(gradient))

def save_gradcam(filename, gcam, raw_image, paper_cmap=False):
    gcam = gcam.cpu().numpy()
    cmap = cm.jet_r(gcam)[..., :3] * 255.0
    if paper_cmap:
        alpha = gcam[..., None]
        gcam = alpha * cmap + (1 - alpha) * raw_image
    else:
        gcam = (cmap.astype(np.float) + raw_image.astype(np.float)) / 2
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

@click.group()
@click.pass_context
def main(ctx):
    print("Mode:", ctx.invoked_subcommand)
    #torch.cuda.empty_cache()

@main.command()
#@click.option("-i", "--images_folder_path", type=str, multiple=True, required=True)
@click.option("-i", "--images_folder_path", type=str, required=True)
#@click.option("-c", "--claseToEval", type=int, required=True)
@click.option("-o", "--output-dir", type=str, default="./results")
@click.option("--cuda/--cpu", default=True)
#def demo0(image_paths, claseToEval, output_dir, cuda):
def demo0(images_folder_path, output_dir, cuda):
    print("Demo-0 running")
    device = get_device(cuda)
    num_classes=4
    targeted_layers = ["avgpool"]
    # Model
    #Instruction to read a model from a "xxx.ckpt" file 
    #model = AlexnetModel(hparams={"lr": 0.00005}, num_classes=4, pretrained=False, seed=None) #seed=manualSeed)   #<<<<<<<<<<<<<<<<-----<<<<----<<<---
    #model = Vgg16Model(hparams={"lr": 0.00005}, num_classes=4, pretrained=False, seed=None)
    #model = Vgg16Model(hparams={}, num_classes=4, pretrained=False, seed=None)
    model = models.vgg16()
    model.classifier = torch.nn.Sequential(torch.nn.Linear(25088, 4096), torch.nn.ReLU(), torch.nn.Dropout(p=0.5), torch.nn.Linear(4096, 256), torch.nn.ReLU(), torch.nn.Dropout(p=0.5), torch.nn.Linear(256, num_classes))
    NameModelLoaded = "vgg16_4c_combined.ckpt"
    model_loaded = torch.load("{}".format(NameModelLoaded))
    #print (">>>>>>>>>print of the loading the vgg16_6Classes.ckpt model <<<<<<<<<<<<")
    for l in model_loaded:
        #print(l)
        if l == "state_dict":
            corrected_dict = {}
            for layerName in model_loaded["state_dict"]:
                NewLayerName = layerName.split(".", 1)[1]
                #print(NewLayerName)
                corrected_dict[NewLayerName] = model_loaded["state_dict"][layerName]
            print (">>>>>>>>> Loading the state_dict of {} <<<<<<<<<<<<".format(NameModelLoaded))
            #print(corrected_dict)
            model.load_state_dict(corrected_dict)
    del model_loaded
    del corrected_dict
    del layerName
    del NewLayerName
    del NameModelLoaded

    model.to(device)
    model.eval()
    # The ... residual layers
    #target_layers = ["classifier.6", "classifier", "avgpool", "features", "features.30", "features.20", "features.10", "features.0"]
    #--->>>target_layers = ["vgg16.avgpool"]#,"vgg16.features.28","vgg16.features.0"]
    #target_layers = ["alex.features.12", "alex.features", "alex.avgpool"]
    #--->>>classes =[claseToEval]#,1,2,3]   # "ACIDE = 0", "Brhusite = 1", "Weddellite = 2", "Whewellite = 3"?
    #folderPath = '{}/*.png'.format(images_folder_path)
    #print(folderPath)
    for filename in glob.glob('{}/*.png'.format(images_folder_path)):
        print("ImgName: {}, ImgType:{}".format(images_folder_path, type(images_folder_path))  )
        #Image_Name = images_folder_path[0].split("\\")[-1]
        #Image_Name = filename.split("/")[-1]
        Image_Name = filename.split("\\")[-1]
        Image_Name = Image_Name.split(".",1)[0]
        print("File_Name: {}".format(filename) )
        print("CutImgName: {}, CutImgType:{}".format(Image_Name, type(Image_Name))  )
        dir = [filename]
        # Images  
        #print("image_paths:{}".format(image_paths))
        images, raw_images = load_images(dir)
        images = torch.stack(images).to(device)
        #print(    "Images lenght: {}".format(len(images) )   )
        gcam = GradCAM(model=model)#, target_layer= targeted_layers) #Suponiendo que se retendran menos "hooks"
        probs, ids = gcam.forward(images)
        for claseToEval in range(4):
            #claseToEval = claseToEval+1
            #print( "Class:{}".format(claseToEval) )
            for target_layer in targeted_layers:
                #for target_class in classes:
                ids_ = torch.LongTensor([[claseToEval]] * len(images)).to(device)    #target_class]] * len(images)).to(device)
                gcam.backward(ids=ids_)
                #    for target_layer in target_layers:
                #print("Generating Grad-CAM @{}".format(target_layer))
                #print("Generating Grad-CAM vgg16.avgpool")
                # Grad-CAM
                #target_layer = "vgg16.avgpool"

                regions = gcam.generate(target_layer= target_layer)  
                #        for j in range(len(images)):
                save_gradcam(
                    filename=osp.join(
                        output_dir,
                        #"{}_{}_Class-{}_({:.5f}).png".format(
                        "VGG16-avgpool--{}--C{}({:.5f}).png".format(
                            Image_Name, claseToEval, float(probs[ids == claseToEval])
                        ),
                    ),
                    #gcam=regions[j, 0],
                    gcam=regions[0, 0],
                    #raw_image=raw_images[j],
                    raw_image=raw_images[0],
                )
            #del gcam.backward(ids=ids_)
        gcam.remove_hook()
        del gcam
        del images
        del raw_images
        del probs
        del ids
        del ids_
        del regions
    ##############################
    #End of 3 FORs   #############
    ##############################
    del model
    #with torch.no_grad():
    #    torch.cuda.empty_cache()
    torch.cuda.empty_cache()
    #print("          -------->>><<<<-----------")
    #print("----------      Fin de Demo0        ")
    #print("          -------->>><<<<-----------")


#@main.command()
#@click.option("-i", "--image-paths", type=str, multiple=True, required=True)
#@click.option("-c", "--claseToEval", type=int, required=True)
#@click.option("-o", "--output-dir", type=str, default="./results")
#@click.option("--cuda/--cpu", default=True)
def demo7(image_paths, claseToEval, output_dir, cuda):
#def demo7(image_paths, output_dir, cuda):
    device = get_device(cuda)
    #print("ImgName: {}, ImgType:{}".format(image_paths, type(image_paths))  )
    Image_Name = image_paths[0].split("\\")[-1]
    Image_Name = Image_Name.split(".")[0]
    #print("CutImgName: {}, CutImgType:{}".format(Image_Name, type(Image_Name))  )

    # Model
    #Instruction to read a model from a "xxx.ckpt" file 
    #model = AlexnetModel(hparams={"lr": 0.00005}, num_classes=4, pretrained=False, seed=None) #seed=manualSeed)   #<<<<<<<<<<<<<<<<-----<<<<----<<<---
    #model = Vgg16Model(hparams={"lr": 0.00005}, num_classes=4, pretrained=False, seed=None)
    #model = Vgg16Model(hparams={}, num_classes=4, pretrained=False, seed=None)
    num_classes=4
    model = models.vgg16()
    model.classifier = torch.nn.Sequential(torch.nn.Linear(25088, 4096), torch.nn.ReLU(), torch.nn.Dropout(p=0.5), torch.nn.Linear(4096, 256), torch.nn.ReLU(), torch.nn.Dropout(p=0.5), torch.nn.Linear(256, num_classes))
    NameModelLoaded = "vgg16_4c_combined.ckpt"
    model_loaded = torch.load("{}".format(NameModelLoaded))
    #print (">>>>>>>>>print of the loading the vgg16_6Classes.ckpt model <<<<<<<<<<<<")
    for l in model_loaded:
        #print(l)
        if l == "state_dict":
            corrected_dict = {}
            for layerName in model_loaded["state_dict"]:
                NewLayerName = layerName.split(".", 1)[1]
                #print(NewLayerName)
                corrected_dict[NewLayerName] = model_loaded["state_dict"][layerName]
            #print (">>>>>>>>> Now: loading the state_dict <<<<<<<<<<<<")
            #print(corrected_dict)
            model.load_state_dict(corrected_dict)
    del model_loaded
    del corrected_dict
    del layerName
    del NewLayerName
    del NameModelLoaded
    #print("Falta de VRAM en GPU???:")
    #torch.cuda.empty_cache()
    #t = torch.cuda.get_device_properties(0).total_memory
    #print("Total     GPU-VRAM:{}".format(t))
    #r = torch.cuda.memory_reserved(0) 
    #a = torch.cuda.memory_allocated(0)
    #f = r-a  # free inside reserved
    #print("reserved  GPU-VRAM:{}".format(r))
    #print("allocated GPU-VRAM:{}".format(a))
    #print("before loading model, free GPU-VRAM: {}".format(f))
    model.to(device)
    #print("xxxxxxxxxxxxxxxxxxx>>>>>>>>>>>>> Falta de VRAM en GPU???: <<<<<<<<<<<<")
    #r = torch.cuda.memory_reserved(0) 
    #a = torch.cuda.memory_allocated(0)
    #f = r-a  # free inside reserved
    #print("reserved  GPU-VRAM:{}".format(r))
    #print("allocated GPU-VRAM:{}".format(a))
    #print("model loaded but not evaluated, free GPU-VRAM: {}".format(f))
    model.eval()
    #r = torch.cuda.memory_reserved(0) 
    #a = torch.cuda.memory_allocated(0)
    #f = r-a  # free inside reserved
    #print("after model evaluation, free GPU-VRAM: {}".format(f))

    # The ... residual layers
    #target_layers = ["classifier.6", "classifier", "avgpool", "features", "features.30", "features.20", "features.10", "features.0"]
    
    #--->>>target_layers = ["vgg16.avgpool"]#,"vgg16.features.28","vgg16.features.0"]
    #target_layers = ["alex.features.12", "alex.features", "alex.avgpool"]
    #--->>>classes =[claseToEval]#,1,2,3]   # "ACIDE = 0", "Brhusite = 1", "Weddellite = 2", "Whewellite = 3"?
    #classes =[0,1,2,3]
    # Images  
    #print("image_paths:{}".format(image_paths))
    images, raw_images = load_images(image_paths)
    images = torch.stack(images).to(device)
    #print(    "Images lenght: {}".format(len(images) )   )
    gcam = GradCAM(model=model)
    probs, ids = gcam.forward(images)
    #for target_class in classes:
    ids_ = torch.LongTensor([[                 claseToEval]] * len(images)).to(device)    #target_class]] * len(images)).to(device)
    gcam.backward(ids=ids_)
    #    for target_layer in target_layers:
    #print("Generating Grad-CAM @{}".format(target_layer))
    #print("Generating Grad-CAM vgg16.avgpool")
            # Grad-CAM
    #target_layer = "vgg16.avgpool"
    target_layer = "avgpool"
    regions = gcam.generate(target_layer= target_layer)  
    #        for j in range(len(images)):
                #print(
                #    "\t#{}: {} ({:.5f})".format(
                #        j, target_class, float(probs[ids == target_class])
                #    )
                #)
    save_gradcam(
        filename=osp.join(
            output_dir,
            #"{}_{}_Class-{}_({:.5f}).png".format(
            "VGG16_avgpool-{}-Class{}({:.5f}).png".format(
                #NameModelLoaded, target_layer, target_class, float(probs[ids == target_class])
                Image_Name, claseToEval, float(probs[ids == claseToEval])
            ),
        ),
        #gcam=regions[j, 0],
        gcam=regions[0, 0],
        #raw_image=raw_images[j],
        raw_image=raw_images[0],
    )
    gcam.remove_hook()
    del gcam
    del model
    #del model_loaded
    #del images
    #del raw_images
    #del probs
    #del ids
    #del ids_
    #del regions
    #with torch.no_grad():
    #    torch.cuda.empty_cache()
    torch.cuda.empty_cache()
    #print("          -------->>><<<<-----------")
    #print("----------      Fin de Demo7        ")
    #print("          -------->>><<<<-----------")


@main.command()
#@click.option("-i", "--image-paths", type=str, multiple=True, required=True)
#@click.option("-c", "--claseToEval", required=True)
#@click.option("-o", "--output-dir", type=str, default="./results")
#@click.option("--cuda/--cpu", default=True)
def demo8():
    print("Demo8 running")
    for filename in glob.glob('samples/*.png'):
        print("----------xxxxxxxxxxxxxxx-----------:")
        for claseToEval in range(4):
            #claseToEval = claseToEval+1
            #t = torch.cuda.get_device_properties(0).total_memory
            #print("Total     GPU-VRAM:{}".format(t))
            #r = torch.cuda.memory_reserved(0) 
            #a = torch.cuda.memory_allocated(0)
            #f = r-a  # free inside reserved
            #print("BEFORE demo7 free GPU-VRAM: {}".format(f))
            #print("reserved  GPU-VRAM:{}".format(r))
            #print("allocated GPU-VRAM:{}".format(a))
            #print( "Class:{}".format(claseToEval) )
            dir = [filename]
            demo7(dir, claseToEval, output_dir="./results", cuda=True)
            torch.cuda.empty_cache()
            #r = torch.cuda.memory_reserved(0) 
            #a = torch.cuda.memory_allocated(0)
            #f = r-a  # free inside reserved
            #print("after demo7 free GPU-VRAM: {}".format(f))
            #print("reserved  GPU-VRAM:{}".format(r))
            #print("allocated GPU-VRAM:{}".format(a))
            
            print("filename:{} Class:{}".format(filename, claseToEval))
            #demo7("{}".format(filename), claseToEval, output_dir="./results", cuda=True)
        #print("filename:{} ".format(filename))
        #dir = [filename]
        #demo7(dir, output_dir="./results", cuda=True)
        #torch.cuda.empty_cache()

@main.command()
@click.option("-i", "--image-path", type=str, multiple=True, required=True)
#@click.option("-c", "--claseToEval", type=int, required=True)
@click.option("-o", "--output-dir", type=str, default="./results")
@click.option("--cuda/--cpu", default=True)
def demo9(image_path, output_dir, cuda):
#def demo7(image_paths, output_dir, cuda):
    """
    Generate Grad-CAM at different layers
    """
    device = get_device(cuda)

    #model = Model() # construct a new model
    #model = models.vgg16()
    #model = models.vgg16(pretrained=True) #original line to read a model from pytorch library (online)
    #model = torch.nn.DataParallel(model)

    # Model
    #Instruction to read a model from a "xxx.ckpt" file 
    #model = AlexnetModel(hparams={"lr": 0.00005}, num_classes=4, pretrained=False, seed=None) #seed=manualSeed)   #<<<<<<<<<<<<<<<<-----<<<<----<<<---
    model = Vgg16Model(hparams={"lr": 0.00005}, num_classes=4, pretrained=False, seed=None)
    #model = Vgg16Model(hparams={}, num_classes=4, pretrained=False, seed=None)
    NameModelLoaded = "vgg16_4c_combined.ckpt"
    model_loaded = torch.load("{}".format(NameModelLoaded))
    #print (">>>>>>>>>print of the loading the vgg16_6Classes.ckpt model <<<<<<<<<<<<")
    for l in model_loaded:
        #print(l)
        if l == "state_dict":
            #print (">>>>>>>>> Now: loading the state_dict <<<<<<<<<<<<")
            model.load_state_dict(model_loaded["state_dict"])
    #print("Falta de VRAM en GPU???:")
    #torch.cuda.empty_cache()
    #t = torch.cuda.get_device_properties(0).total_memory
    #print("Total     GPU-VRAM:{}".format(t))
    #r = torch.cuda.memory_reserved(0) 
    #a = torch.cuda.memory_allocated(0)
    #f = r-a  # free inside reserved
    #print("reserved  GPU-VRAM:{}".format(r))
    #print("allocated GPU-VRAM:{}".format(a))
    #print("before loading model, free GPU-VRAM: {}".format(f))

    model.to(device)

    #print("xxxxxxxxxxxxxxxxxxx>>>>>>>>>>>>> Falta de VRAM en GPU???: <<<<<<<<<<<<")
    #r = torch.cuda.memory_reserved(0) 
    #a = torch.cuda.memory_allocated(0)
    #f = r-a  # free inside reserved
    #print("reserved  GPU-VRAM:{}".format(r))
    #print("allocated GPU-VRAM:{}".format(a))
    #print("model loaded but not evaluated, free GPU-VRAM: {}".format(f))
    model.eval()
    #r = torch.cuda.memory_reserved(0) 
    #a = torch.cuda.memory_allocated(0)
    #f = r-a  # free inside reserved
    #print("after model evaluation, free GPU-VRAM: {}".format(f))

    # The ... residual layers
    #target_layers = ["classifier.6", "classifier", "avgpool", "features", "features.30", "features.20", "features.10", "features.0"]
    
    #--->>>target_layers = ["vgg16.avgpool"]#,"vgg16.features.28","vgg16.features.0"]
    #target_layers = ["alex.features.12", "alex.features", "alex.avgpool"]
    #--->>>classes =[claseToEval]#,1,2,3]   # "ACIDE = 0", "Brhusite = 1", "Weddellite = 2", "Whewellite = 3"?
    #classes =[0,1,2,3]
    classes = 3
    # Images  
    #print("image_paths:{}".format(image_paths))
    images, raw_images = load_images(image_path)
    Img_name = image_path[0].split("\\")
    Img_name = Img_name[2].split(".")[0]
    #print(Img_name)
    images = torch.stack(images).to(device)
    #print(    "Images lenght: {}".format(len(images) )   )
    gcam = GradCAM(model=model)
    probs, ids = gcam.forward(images)
    #for target_class in classes:
    ids_ = torch.LongTensor([[classes]] * len(images)).to(device)    #target_class]] * len(images)).to(device)
    gcam.backward(ids=ids_)
    #    for target_layer in target_layers:
    #print("Generating Grad-CAM @{}".format(target_layer))
    #print("Generating Grad-CAM vgg16.avgpool")
            # Grad-CAM
    regions = gcam.generate(target_layer=       "vgg16.avgpool") #target_layer)
    #        for j in range(len(images)):
                #print(
                #    "\t#{}: {} ({:.5f})".format(
                #        j, target_class, float(probs[ids == target_class])
                #    )
                #)
    save_gradcam(
        filename=osp.join(
            output_dir,
            #"{}_{}_Class-{}_({:.5f}).png".format(
            "{}_last-avgpool_Class{}_({:.5f})_{}.png".format(
                #NameModelLoaded, target_layer, target_class, float(probs[ids == target_class])
                NameModelLoaded, classes, float(probs[ids == classes]), Img_name
            ),
        ),
        #gcam=regions[j, 0],
        gcam=regions[0, 0],
        #raw_image=raw_images[j],
        raw_image=raw_images[0],
    )
    gcam.remove_hook()
    torch.cuda.empty_cache()
    #print("          -------->>><<<<-----------")
    #print("----------      Fin de Demo9        ")
    #print("          -------->>><<<<-----------")

if __name__ == "__main__":
    torch.cuda.empty_cache()
    main()
    torch.cuda.empty_cache()
