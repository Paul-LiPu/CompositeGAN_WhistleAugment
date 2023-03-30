from __future__ import print_function
import torch
import numpy as np
from PIL import Image
import os
from torch import is_tensor
from torch.autograd import Variable


# Converts a Tensor into a float
def tensor2float(input_error):
    if is_tensor(input_error):
        if input_error.dim() < 1:
            error = float(input_error)
        else:
            error = input_error[0]
    elif isinstance(input_error, Variable):
        error = input_error.data[0]
    else:
        error = input_error
    return error


# Converts a Tensor into an image array (numpy)
# |imtype|: the desired type of the converted numpy array
def tensor2im(input_image, imtype=np.uint8):
    if is_tensor(input_image):
        image_tensor = input_image
    elif isinstance(input_image, Variable):
        image_tensor = input_image.data
    else:
        return input_image

    # image_numpy = image_tensor.cpu().float().numpy()
    image_numpy = image_tensor.detach().cpu().float().numpy()
    shape = image_numpy.shape
    image_numpy = image_numpy.reshape(shape[0], shape[2], shape[3])
    if image_numpy.shape[0] > 1:
        image_numpy = save_images(image_numpy)
        image_numpy = image_numpy[np.newaxis, ]

    if image_numpy.shape[0] == 1:
        image_numpy = np.tile(image_numpy, (3, 1, 1))
    image_numpy = (np.transpose(image_numpy, (1, 2, 0)) + 1) / 2.0 * 255.0
    return image_numpy.astype(imtype)


def save_images(X):
    # [0, 1] -> [0,255]
    # if isinstance(X.flatten()[0], np.floating):
    #     X = (255.99*X).astype('uint8')

    n_samples = X.shape[0]
    rows = int(np.sqrt(n_samples))
    while n_samples % rows != 0:
        rows -= 1

    nh, nw = rows, n_samples//rows

    if X.ndim == 2:
        X = np.reshape(X, (X.shape[0], int(np.sqrt(X.shape[1])), int(np.sqrt(X.shape[1]))))

    if X.ndim == 4:
        # BCHW -> BHWC
        X = X.transpose(0,2,3,1)
        h, w = X[0].shape[:2]
        img = np.zeros((h*nh, w*nw, 3))
    elif X.ndim == 3:
        h, w = X[0].shape[:2]
        img = np.zeros((h*nh, w*nw))

    for n, x in enumerate(X):
        j = n//nw
        i = n%nw
        img[j*h:j*h+h, i*w:i*w+w] = x

    return img

def diagnose_network(net, name='network'):
    mean = 0.0
    count = 0
    for param in net.parameters():
        if param.grad is not None:
            mean += torch.mean(torch.abs(param.grad.data))
            count += 1
    if count > 0:
        mean = mean / count
    print(name)
    print(mean)


def save_image(image_numpy, image_path):
    image_pil = Image.fromarray(image_numpy)
    image_pil.save(image_path)


def print_numpy(x, val=True, shp=False):
    x = x.astype(np.float64)
    if shp:
        print('shape,', x.shape)
    if val:
        x = x.flatten()
        print('mean = %3.3f, min = %3.3f, max = %3.3f, median = %3.3f, std=%3.3f' % (
            np.mean(x), np.min(x), np.max(x), np.median(x), np.std(x)))


def mkdirs(paths):
    if isinstance(paths, list) and not isinstance(paths, str):
        for path in paths:
            mkdir(path)
    else:
        mkdir(paths)


def mkdir(path):
    if not os.path.exists(path):
        os.makedirs(path)





import numpy as np
import os
import glob
import torch
import re

####################### File System ###############################
def readLines(file):
    """
    Read all lines in a file , and return the lines without linefeed using a list.
    :param file: path the file
    :return: list of strings, each string is a line in the file
    """
    with open(file) as f:
        data = f.readlines()
    data = [line.strip() for line in data]
    return data

def checkDir(dir):
    if not os.path.exists(dir):
        os.makedirs(dir)

def checkDirs(dirs):
    for dir in dirs:
        checkDir(dir)

def globx(dir, patterns):
    result = []
    for pattern in patterns:
        subdirs = glob.glob(dir + '/' + pattern)
        result.extend(subdirs)
    result = list(set(result))
    list.sort(result)
    return result

def globxx(dir, extensions):
    images = []
    assert os.path.isdir(dir), '%s is not a valid directory' % dir

    for root, _, fnames in sorted(os.walk(dir)):
        for fname in fnames:
            if any(fname.endswith(extension) for extension in extensions):
                path = os.path.join(root, fname)
                images.append(path)
    return images

def globxxx(dir, fragments):
    images = []
    assert os.path.isdir(dir), '%s is not a valid directory' % dir

    for root, _, fnames in sorted(os.walk(dir)):
        for fname in fnames:
            if all(fragment in fname for fragment in fragments):
                path = os.path.join(root, fname)
                images.append(path)
    return images



####################### Image Processing ###############################
def patch2image(X):
    """
    Stitch a batch of patches together into one image.
    :param X: input patches, in range[0,1](float) or [0,255](uint8)
    :shape: BxCxHxW or BxHxW or BxHW(if H=W)
    :return: stitched image.
    """

    # [0, 1] -> [0,255]

    n_samples = X.shape[0]
    rows = int(np.sqrt(n_samples))
    while n_samples % rows != 0:
        rows -= 1

    nh, nw = rows, int(n_samples/rows)

    # if the patch is flattened, unflat it
    if X.ndim == 2:
        X = np.reshape(X, (X.shape[0], int(np.sqrt(X.shape[1])), int(np.sqrt(X.shape[1]))))

    # construct the stitching image
    if X.ndim == 4:
        # BCHW -> BHWC
        X = X.transpose(0, 2, 3, 1)
        h, w = X[0].shape[:2]
        img = np.zeros((h * nh, w * nw, 3))
    elif X.ndim == 3:
        h, w = X[0].shape[:2]
        img = np.zeros((h * nh, w * nw)).astype(X.dtype)

    for n, x in enumerate(X):
        j = int(n / nw)
        i = n % nw
        img[j * h: j * h + h, i * w: i * w + w] = x

    return img




def get_image_patch(image, pts):
    """
    return image patches from images.
    :param image: images to extract patches, could be BxCxHxW
    :param pts: patch vertex location
    :return:
    """
    return image[..., int(pts[1]):int(pts[5]), int(pts[0]):int(pts[4])]


def im_unit8(images, lb=0, ub=1):
    """
    convert numpy img from any range to [0, 255] and unit8
    :param images: input images
    :param lb: lower bound of input images pixel values
    :param ub: upper bound of input images pixel values
    :return: an unit8 image which is ready for
    """
    if images.dtype == np.uint8:
        return images
    images = np.clip((images - lb) * 1.0 / (ub - lb) * 255, 0, 255).astype('uint8')
    return images

def im_float32(images, lb=0, ub=255):
    """
    convert numpy img from any range to [0, 1] and float
    :param images: input images
    :param lb: lower bound of input images pixel values
    :param ub: upper bound of input images pixel values
    :return: an unit8 image which is ready for
    """
    images = images.astype('float32')
    images = (images - lb) * 1.0 / (ub - lb)
    images = images.astype('float32')
    return images

def im_float32_symm(images, lb=0, ub=255):
    """
    convert numpy img from any range to [-1, 1] and float
    :param images: input images
    :param lb: lower bound of input images pixel values
    :param ub: upper bound of input images pixel values
    :return: an unit8 image which is ready for
    """
    images = images.astype('float32')
    images = (images - (lb + ub) / 2.0) * 2.0 / (ub - lb)
    images = images.astype('float32')
    return images

def im_min_max_norm(img):
    min_v = np.min(img)
    max_v = np.max(img)
    if max_v == min_v:
        img[...] = 1
    else:
        img = (img - min_v) / (max_v - min_v)
    return img

def select_patch(img_size, patch_size, batchsize=32, mode='random'):
  if mode == 'random':
    x = np.random.randint(0, img_size[1] - patch_size + 1, size=(batchsize))
    y = np.random.randint(0, img_size[0] - patch_size + 1, size=(batchsize))
  elif mode == 'top_left':
    x = 0
    y = 0
  elif mode == 'bottom_left':
    x = 0
    y = img_size[0] - patch_size - 1
  elif mode == 'top_right':
    x = img_size[1] - patch_size - 1
    y = 0
  elif mode == 'bottom_right':
    x = img_size[1] - patch_size - 1
    y = img_size[0] - patch_size - 1
  elif mode == 'center':
    x = int((img_size[1] - patch_size) / 2.0)
    y = int((img_size[0] - patch_size) / 2.0)
  else:
    print('Patch Selection mode not implemented')
    exit(-1)

  bl_x = patch_size + x
  bl_y = y.copy()
  br_x = patch_size + x
  br_y = patch_size + y
  tr_x = x.copy()
  tr_y = patch_size + y

  return np.stack([x, y, bl_x, bl_y, br_x, br_y, tr_x, tr_y], axis=1)

####################### Video Processing ###############################




####################### Audio Processing ###############################



####################### PyTorch ######################################
def worker_init_fn(worker_id):
    np.random.seed((torch.initial_seed() + worker_id) % (2 ** 32))

# def find_last_model(model_dir, model_name):
#     models = globx(model_dir, [model_name + '-last*.pth'])
#     if len(models) > 0:
#         last_model =  models[0]
#     else:
#         models = globx(model_dir, [model_name + '-iter*.pth'])
#         model_iters = [model.split('-')[-1] for model in models]
#         model_iters = [int(model.split('.')[0]) for model in model_iters]
#         idx = np.argsort(model_iters)
#         last_model = models[idx[-1]]
#     return last_model

def find_last_model(model_dir, model_name):
    # models = globx(model_dir, [model_name + '*.pth'])
    models = globx(model_dir, [model_name + '*'])
    latest_model = max(models, key=os.path.getctime)
    return latest_model

def find_first_model(model_dir, model_name):
    models = globx(model_dir, [model_name + '*.pth'])
    first_model = min(models, key=os.path.getctime)
    return first_model

def get_model_name(model_name, iter, itername='iter'):
    return model_name + '-' + itername + '-' + str(iter) + '.pth'

def resume_model(model, model_dir, model_name):
    last_model = find_last_model(model_dir, model_name)
    niter, nepoch = load_model(model, last_model)
    return last_model, niter, nepoch

def load_part_of_model(model, model_dir, model_name):
    last_model = find_last_model(model_dir, model_name)
    dict_in_file = torch.load(last_model)
    model = load_part_model(model, dict_in_file['model'])
    return last_model, dict_in_file['n_iter'], dict_in_file['n_epoch']

def load_part_model(model, pretrained_dict):
    model_dict = model.state_dict()
    # 1. filter out unnecessary keys
    pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
    # 2. overwrite entries in the existing state dict
    model_dict.update(pretrained_dict)
    # 3. load the new state dict
    model.load_state_dict(model_dict)
    return model

def load_part_weights(model, weights_file):
    model_dict = model.state_dict()
    dict_in_file = torch.load(weights_file)
    pretrained_dict = dict_in_file['model']
    # 1. filter out unnecessary keys
    pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
    # 2. overwrite entries in the existing state dict
    model_dict.update(pretrained_dict)
    # 3. load the new state dict
    model.load_state_dict(model_dict)
    return dict_in_file['n_iter'], dict_in_file['n_epoch']

def save_model(model, iter, epoch):
    return {'model': model.state_dict(), 'n_iter':iter, 'n_epoch': epoch}

def load_model(model, weights_file):
    dict_in_file = torch.load(weights_file)
    model.load_state_dict(dict_in_file['model'])
    return dict_in_file['n_iter'], dict_in_file['n_epoch']

def check_num_model(model_dir, model_name, num):
    models = globx(model_dir, [model_name + '*.pth'])
    return len(models) >= num

def rm_suffix(filename):
    temp = filename.split('.')
    result = '.'.join(temp[:-1])
    return result

def read_array_file(file, type):
    data = readLines(file)
    data = [x.split() for x in data]
    data = np.array(data).astype(type)
    return data

def get_column(data, col, type):
    data = [type(item) for item in data[:, col]]
    return np.squeeze(np.asarray(data))


def sort_frames(imgs):
    imgnames = [os.path.basename(img) for img in imgs]
    frame_num = []
    for imgname in imgnames:
        temp = imgname.split('frame')
        temp = temp[1]
        temp = temp.split('.')
        temp = temp[0]
        temp = temp.split('_')
        temp = temp[0]
        frame_n = int(temp)
        frame_num.append(frame_n)

    sort_idx = np.argsort(frame_num)
    imgs = [imgs[i] for i in list(sort_idx)]

    return imgs


def sort_file(files, pattern, type=int):
    filenames = [os.path.basename(file) for file in files]
    nums = []
    matcher = re.compile(pattern)
    for filename in filenames:
        results = matcher.findall(filename)
        num = type(results[0])
        nums.append(num)

    sort_idx = np.argsort(nums)
    files = [files[i] for i in list(sort_idx)]
    return files, sort_idx

########################### Math ##################################
def nearest_neighbor(vec1, vec2):
    dists = []
    idx1 = 0
    idxs = []
    for v1 in vec1:
        idx2 = 0
        for v2 in vec2:
            dist = vector_length(v1 -v2)
            dists.append(dist)
            idxs.append([idx1, idx2])
            idx2 += 1
        idx1 += 1

    min_idx = np.argmin(dists)
    min_dist = dists[min_idx]
    min_v1 = vec1[idxs[min_idx][0]]
    min_v2 = vec2[idxs[min_idx][1]]
    return min_dist, min_v1, min_v2


def vector_length(v):
    return np.sqrt(np.sum(v ** 2))


def condition_number(matrix):
    """
        Compute the condition number of a matrix
        :param matrix: matrix have shape m x n
        :return: condition number by L2-norm of matrix
        """
    cond_mat = torch.matmul(matrix, torch.transpose(matrix, 0, 1))  # (m, n) * (n, m) = (m, m)
    eigenvalue, eigenvectors = torch.symeig(cond_mat, eigenvectors=True)
    max_eigenvalue = torch.max(eigenvalue)
    min_eigenvalue = torch.min(eigenvalue)
    return torch.sqrt((max_eigenvalue + 1e-4) / (min_eigenvalue + 1e-4))
