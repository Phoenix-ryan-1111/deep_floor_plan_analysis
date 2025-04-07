import torch
from model import FloorPlanNet
from PIL import Image
from torchvision import transforms
import numpy as np
import matplotlib.pyplot as plt

# Color maps (from rgb_ind_convertor.py)
floorplan_map = {
    0: [255, 255, 255],  # background
    1: [192, 192, 224],  # closet
    2: [192, 255, 255],  # batchroom/washroom
    3: [224, 255, 192],  # livingroom/kitchen/dining room
    4: [255, 224, 128],  # bedroom
    5: [255, 160, 96],  # hall
    6: [255, 224, 224],  # balcony
    7: [255, 255, 255],  # not used
    8: [255, 255, 255],  # not used
    9: [255, 60, 128],  # door & window
    10: [0, 0, 0]  # wall
}


def ind2rgb(ind_im, color_map=floorplan_map):
    rgb_im = np.zeros((ind_im.shape[0], ind_im.shape[1], 3), dtype=np.uint8)
    for i, rgb in color_map.items():
        rgb_im[(ind_im == i)] = rgb
    return rgb_im


def predict(image_path, model_path='floorplan_model.pth'):
    # Load model
    model = FloorPlanNet()
    model.load_state_dict(torch.load(model_path))
    model.eval()

    # Load and preprocess image
    transform = transforms.Compose([
        transforms.Resize((512, 512)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

    image = Image.open(image_path).convert('RGB')
    input_tensor = transform(image).unsqueeze(0)

    # Predict
    with torch.no_grad():
        room_pred, boundary_pred = model(input_tensor)

        # Convert to numpy
        room_pred = torch.argmax(room_pred, dim=1).squeeze().cpu().numpy()
        boundary_pred = torch.argmax(boundary_pred,
                                     dim=1).squeeze().cpu().numpy()

    # Merge results
    floorplan = room_pred.copy()
    floorplan[boundary_pred == 1] = 9
    floorplan[boundary_pred == 2] = 10

    # Convert to RGB
    floorplan_rgb = ind2rgb(floorplan)

    # Display results
    plt.figure(figsize=(12, 6))
    plt.subplot(121)
    plt.imshow(image)
    plt.title('Input Image')
    plt.subplot(122)
    plt.imshow(floorplan_rgb)
    plt.title('Prediction')
    plt.show()


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--im_path',
                        type=str,
                        default='./demo/45765448.jpg',
                        help='input image paths.')
    args = parser.parse_args()
    predict(args.im_path)
