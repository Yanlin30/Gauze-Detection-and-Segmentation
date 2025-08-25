import os, torch, cv2, json
import numpy as np
from PIL import Image
import torch.utils.data as td
from torchvision import transforms
from sklearn.model_selection import KFold
from torch.utils.data import DataLoader
device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")

def get_data(log_file, mask_root, task):
    """

    :param log_file: 保存有图片名称的txt文件
    :param mask_root:
    :param tasl: seg or detection
    :return: 返回存有图像和mask路径的两个列表
    """
    with open(log_file, 'r') as f:
        lines = f.readlines()

    img_set = []
    mask_set = []
    for line in lines:
        img_name = line.strip().split("/")[-1].split("\\")[-1]
        if img_name not in os.listdir(mask_root):
            if task == 'seg':
                mask_set.append(None)
            else:
                continue
        else:
            mask_set.append(img_name)
        img_set.append(img_name)

    return img_set, mask_set

def seg_loader(data_img, data_mask, img_root, mask_root):
    """
    加载图像
    :param data_img:
    :param data_mask:
    :param img_root:
    :param mask_root:
    :return:
    """
    trans = transforms.Compose([transforms.ToTensor(), transforms.Resize([512, 512]), transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))])

    img_path = os.path.join(img_root, data_img)
    open_img = Image.open(img_path)
    img = open_img.convert("RGB")
    img = np.array(img)

    if data_mask == None:
        img_size = [img.shape[0], img.shape[1]]
        mask = np.zeros(img_size, np.uint8)
    else:
        mask_path = os.path.join(mask_root, data_mask)
        mask = np.array(Image.open(mask_path))

    mask = torch.from_numpy(cv2.resize(mask, (512, 512), interpolation=cv2.INTER_NEAREST) / 255)
    return trans(img), mask


def detection_loader(data_img, data_json, img_root, json_root):
    """

    :param data_img:
    :param data_json:
    :param img_root:
    :param json_root:
    :return:list[dict,...]
    """
    trans = transforms.Compose([transforms.ToTensor(), transforms.Resize([512, 512]),
                                transforms.Normalize([0.48235, 0.45882, 0.40784],
                                                     (0.229, 0.224, 0.225))])

    img_path = os.path.join(img_root, data_img)
    open_img = Image.open(img_path)
    img = open_img.convert("RGB")
    img = np.array(img)

    json_path = os.path.join(json_root, data_json[:-3]+"json")
    with open(json_path, encoding='UTF-8') as f:
        content = json.load(f)

    shapes = content['shapes']
    points = []
    targets = {}
    area = []
    for shape in shapes:
        target = {}
        category = shape['label']
        if "rectangle" in category:
            temp = np.array(shape['points'])
            x, y = img.shape[1], img.shape[0]
            x_min = min(temp[0][0], temp[1][0]) / x * 512
            x_max = max(temp[0][0], temp[1][0]) / x * 512
            y_min = min(temp[0][1], temp[1][1]) / y * 512
            y_max = max(temp[0][1], temp[1][1]) / y * 512
            points.append([x_min, y_min, x_max, y_max])
            area.append((x_max - x_min)*(y_max - y_min))

    n_number = len(points)
    # if n_number == 0:
    #     idx = 0
    # else:
    #     idx = random.randint(0, n_number-1)

    # points = points[idx]
    # area = area[idx]
    # targets["boxes"] = torch.tensor(points, dtype=torch.float32)
    targets["boxes"] = torch.tensor(points, dtype=torch.float32)
    targets["labels"] = torch.tensor([1]*n_number, dtype=torch.long)
    # targets["image_id"] = data_img
    targets["area"] = torch.tensor(area, dtype=torch.float32)

    return trans(img), targets


def my_collate(batch):
    images = []
    targets = []
    for b in batch:
        images.append(b[0])
        for k, v in b[1].items():
            b[1][k] = v.to(device)
        targets.append(b[1])
    images = torch.stack(images, dim=0)
    return images, targets

def refine_targets(targets):
    print(targets)
    new_tagets = []
    n_number = len(targets)
    print(n_number)
    for i in range(n_number):
        target = {}
        target["boxes"] = targets[i].reshape(-1, 4).to(device)
        target["labels"] = targets["labels"].to(device)
        target["image_id"] = targets["image_id"]
        target["area"] = targets["area"].to(device)
        new_tagets.append(target)
    return new_tagets



class seg_dataset(td.Dataset):
    def __init__(self, data_image, data_mask, image_root, mask_root):
        self.image = data_image
        self.img_root = image_root
        self.mask = data_mask
        self.mask_root = mask_root

    def __getitem__(self, index):
        img, mask = seg_loader(data_img=self.image[index], img_root=self.img_root,
                               data_mask=self.mask[index], mask_root=self.mask_root)
        return img, mask

    def __len__(self):
        return len(self.image)


class detection_dataset(td.Dataset):
    def __init__(self, data_image, data_json, image_root, json_root):
        self.image = data_image
        self.json = data_json
        self.img_root = image_root
        self.json_root = json_root

    def __getitem__(self, idx):
        img, target = detection_loader(data_img=self.image[idx], img_root=self.img_root,
                                       data_json=self.json[idx], json_root=self.json_root)
        return img, target

    def __len__(self):
        return len(self.image)

        
