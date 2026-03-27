import pandas as pd
from PIL import Image
from torchvision import transforms
import torch
import numpy as np
import torch 
from torchvision.models import resnet18

class getEmbeddings:
  def __init__(self,path) -> None:
    self.model=resnet18(pretrained=True)
    self.model.fc=torch.nn.Identity()
    self.model.eval()
    self.base=path
  
  def transform(self,img):

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    return transform(img)

  def getEmbeddings(self,img_path):
    path=self.base+img_path
    img=Image.open(path).convert("RGB")
    img = self.transform(img).unsqueeze(0) 
    with torch.no_grad():
      embedding = self.model(img)

    embedding = embedding.squeeze().numpy()
    return embedding/np.linalg.norm(embedding)
  
  def embedder(self,img_path):
    path=img_path
    img=Image.open(path).convert("RGB")
    img = self.transform(img).unsqueeze(0) 
    with torch.no_grad():
      embedding = self.model(img)

    embedding = embedding.squeeze().numpy()
    return embedding/np.linalg.norm(embedding)