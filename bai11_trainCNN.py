import os
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from argparse import ArgumentParser
from bai10_buildCNNmodel import SimpleCNN


def get_args():
    parser = ArgumentParser(description="CNN inference")
    parser.add_argument("--image-path", "-p", type=str, default="cat_01.jpg")
    parser.add_argument("--image-size", "-i", type=int, default=224, help="Image size")
    parser.add_argument("--checkpoint", "-c", type=str, default="trained_models/best_cnn.pt")
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    categories = ["cat", "dog", "cow", "horse", "chicken", "squirrel", "sheep", "elephant", "spider", "butterfly"]
    test_dir = r"C:/Users/User/source/repos/DeepLearningProject/test_images/web_animals_20261016"

    args = get_args()
    if torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
    
    model = SimpleCNN(num_classes=len(categories)).to(device)
    if args.checkpoint:
        checkpoint = torch.load(args.checkpoint, map_location=device)
        model.load_state_dict(checkpoint['model'])
    else: 
        print("NO checkpoint found!")
        exit()
    
    model.eval()
    image_path = os.path.join(test_dir, args.image_path)
    ori_image = cv2.imread(image_path)
    image = cv2.cvtColor(ori_image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, (args.image_size, args.image_size))
    image = np.transpose(image, (2,0,1)) / 255.0
    image = image[None, :, :, :]
    image = torch.from_numpy(image).to(device).float()

    softmax = nn.Softmax(dim=-1)
    with torch.no_grad():
        output = model(image)
        probs = softmax(output)
        print(output.shape, probs.shape)
    
    max_idx = torch.argmax(probs)
    predicted_class = categories[max_idx.item()]
    print("the test image is about {} with confident score of {:.4f}".format(predicted_class, probs[0, max_idx].item()))
    cv2.imshow("{} {:.2f}".format(predicted_class, probs[0, max_idx].item() * 100), ori_image)
    cv2.waitKey(0)
