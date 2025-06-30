import torch
import torchvision.transforms as T
from torchvision.models.segmentation import deeplabv3_mobilenet_v3_large
from PIL import Image
import numpy as np
import cv2
import os
import warnings

INPUT_IMAGE = "p4.png"
OUTPUT_IMAGE = "tests/0.46m2_per_pixel/p4.png"
MODEL_PATH = 'best_model.pth'


def proccess():

    if not os.path.exists(INPUT_IMAGE):
        print(f"Ошибка: Файл {INPUT_IMAGE} не найден!")
        return

    if not os.path.exists(MODEL_PATH):
        print(f"Ошибка: Модель {MODEL_PATH} не найдена!")
        return

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    #print(f'Используется устройство: {device}')

    warnings.filterwarnings("ignore", category=UserWarning, module="torchvision.models._utils")

    model = deeplabv3_mobilenet_v3_large(weights=None, num_classes=1)
    state_dict = torch.load(MODEL_PATH, map_location=device)

    model.load_state_dict(state_dict, strict=False)
    model = model.to(device)
    model.eval()

    transform = T.Compose([
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    original_img = Image.open(INPUT_IMAGE).convert('RGB')
    original_size = original_img.size
    img = original_img.resize((512, 512))
    input_tensor = transform(img).unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(input_tensor)['out']
        prediction = torch.sigmoid(output[0, 0])
        mask = (prediction > 0.27).float().cpu().numpy()


    mask = (mask * 255).astype(np.uint8)
    mask_resized = cv2.resize(mask, original_size, interpolation=cv2.INTER_NEAREST)

    original_np = np.array(original_img)

    bw_mask = np.stack([mask_resized] * 3, axis=-1)

    color_mask = np.zeros((mask_resized.shape[0], mask_resized.shape[1], 3), dtype=np.uint8)
    color_mask[:, :, 2] = mask_resized  # Красный канал
    blended = cv2.addWeighted(original_np, 0.7, color_mask, 0.3, 0)

    collage_width = original_size[0] * 3
    collage_height = original_size[1]
    collage = np.zeros((collage_height, collage_width, 3), dtype=np.uint8)

    collage[:, :original_size[0]] = original_np

    collage[:, original_size[0]:2 * original_size[0]] = bw_mask

    collage[:, 2 * original_size[0]:] = blended

    
    result_img = Image.fromarray(collage)
    result_img.save(OUTPUT_IMAGE)
    print(f'Коллаж сохранен как: {OUTPUT_IMAGE}')



proccess()