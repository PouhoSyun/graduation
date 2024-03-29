# data fetching and preprocessing module

import mat73, cv2, os, math
import numpy as np
import torch
import torch.nn as nn
import torch.utils.data as data
import matplotlib.pyplot as plt
import albumentations as alb

# 4 basic methods for numpy.array -- fundamental stack/split & fine stack/split functions
def hvstack(src: np.ndarray):
    size = int(src.shape[1])
    pieces = int(src.shape[0] ** 0.5)
    lines = []
    for line in src.reshape(pieces, pieces, size, size):
        lines.append(np.hstack(line))
    dst = np.vstack(lines).astype(np.uint8)
    return dst

def hvsplit(src: np.ndarray, pieces = 4):
    lines = np.vsplit(src, pieces)
    dst = []
    for line in lines:
        line = np.hsplit(line, pieces)
        dst.extend(line)
    return np.array(dst)

# fine stack -- array[n*n, size, size] -> array[n*size, n*size]
def fine_stack(src: np.ndarray):
    size = int(src.shape[1])
    pieces = int(src.shape[0] ** 0.5)
    src = hvstack(src)
    dst = []
    for i in range(size * size):
        dst.append(src[(i//size)::size, (i%size)::size].reshape(pieces, pieces))
    return hvstack(np.array(dst))

# fine split -- array[n*size, n*size] -> array[n*n, size, size]
def fine_split(src: np.ndarray, pieces = 4):
    size = int(src.shape[1] / pieces)
    dst = []
    for i in range(pieces * pieces):
        dst.append(src[(i//pieces)::pieces, (i%pieces)::pieces].reshape(size, size))
    return np.array(dst)

# size of event camera is 190*180
# transform matfile to streamfile, then use spatial-temporal voxel grid method to record events.
def pack_event_stream(dataset, file_cnt, time, split=True):
    try:
        event_countmap = np.load("dataset_prep/" + dataset + "/events_voxel.npy")
    except:
        ev_stream = []
        for i in range(1, file_cnt + 1):
            data = mat73.loadmat("test/dataset/" + dataset + "/events_clip/frame" + str(i) + ".mat")
            events = np.array(tuple(zip(data['clipev']['ev_x'],
                                data['clipev']['ev_y'],
                                data['clipev']['ev_p'],
                                data['clipev']['ev_t'])))
            ev_stream.append(events)

        event_countmap = []
        for events in ev_stream:
            # DAVIS infrared senser use linear threshold
            event_field = np.zeros((200, 200))
            for event in events:
                #consider pre/past event affect
                event_field[int(event[1])][int(event[0])] += event[2] / (1 + math.exp(time - event[3])) 
            event_field = np.clip(event_field, -255, 255)
            for it in np.nditer(event_field, op_flags=["readwrite"]):
                it[...] = np.uint8(255 / (1 + math.exp(-it)))

            # split the output into 16 50*50 pieces
            if not split:
                event_countmap.append(event_field)
            else:
                event_countmap.append(hvsplit(event_field))

        event_countmap = np.array(event_countmap)
        if not os.path.exists("dataset_prep/" + dataset):
            os.makedirs("dataset_prep/" + dataset)
        np.save("dataset_prep/" + dataset + "/events_voxel.npy", event_countmap)

    return torch.tensor(event_countmap)

# size of frame camera is 1520 * 1440, then split to 16 400*400 cell-pics.
def load_frame_png(dataset, file_id, cmap, split=True):
    frame = cv2.imread("test/dataset/" + dataset + "/RGB_frame/frame" + str(file_id) + ".png", cmap)
    h_pad = int((1600 - frame.shape[1]) / 2)
    v_pad = int((1600 - frame.shape[0]) / 2)
    frame = cv2.copyMakeBorder(frame, v_pad, v_pad, h_pad, h_pad, cv2.BORDER_CONSTANT, value=(128, 128, 128))
    if not split: return torch.Tensor(frame)
    else:
        lines = np.vsplit(frame, 4)
        frame = []
        for line in lines:
            line = np.hsplit(line, 4)
            frame.extend(line)
        return torch.tensor(frame)

def pack_frame_png(dataset, file_cnt, cmap):
    frames = []
    for iter in range(file_cnt):
        frame = load_frame_png(dataset, iter + 1, cmap)
        frames.append(frame.tolist())
    return torch.tensor(frames)

# concatenate tensors of event and frame
def concat_tensors(dataset, time, cmap):
    iter_size = len(os.listdir("test/dataset/" + dataset + "/RGB_frame"))
    try:
        frame = np.load("dataset_prep/" + dataset + "/frame.npy")
    except:
        frame = pack_frame_png(dataset, iter_size, cmap).tolist()
        np.save("dataset_prep/" + dataset + "/frame.npy", frame)
    
    try:
        events = np.load("dataset_prep/" + dataset + "/events_voxel.npy") 
    except:
        __, events = pack_event_stream(dataset, iter_size, time)
        events = events.tolist()
        np.save("dataset_prep/" + dataset + "/events_voxel.npy", events)
        
    data = list(zip(frame[0:-1], events, frame[1:]))
    return data

# get frame dataset: cmap--cv2.IMREAD_*, size--square edge length of the image
class Frame_Dataset(data.Dataset):
    def __init__(self, dataset, cmap, size):
        path = "test/dataset/" + dataset + "/RGB_frame"

        self.images = [os.path.join(path, file) for file in os.listdir(path)]
        self._length = len(self.images)
        self.cmap = cmap
        
        self.rescaler = alb.SmallestMaxSize(max_size=size)
        self.cropper = alb.CenterCrop(height=size, width=size)
        self.preprocessor = alb.Compose([self.rescaler, self.cropper])
        pass

    def __len__(self):
        return self._length
    
    def __getitem__(self, index):
        item = np.array(cv2.imread(self.images[index], self.cmap)).astype(np.uint8)
        item = self.preprocessor(image=item)["image"]
        # item = (item / 127.5 - 1.0).astype(np.float32)
        # item = item.transpose(2, 0, 1)
        return torch.Tensor(item)

#dataset class for dataloader
class DAVIS_Dataset(data.Dataset):
    def __init__(self, dataset, time, cmap, data_type):
        self.data = concat_tensors(dataset, time, cmap)
        self.data_type = data_type
        pass

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        item = self.data[index]
        # return torch.Tensor(item[0]), torch.Tensor(item[1]), torch.Tensor(item[2])
        return torch.Tensor(item[self.data_type])
    
    def __show__(self, index):
        frame, events, target = self.__getitem__(index)
        cv2.namedWindow("Dataset Preview", cv2.WINDOW_FREERATIO)
        stack_frame = hvstack(frame)
        stack_events = cv2.copyMakeBorder(hvstack(events), 700, 700, 0, 0, cv2.BORDER_CONSTANT, value=(128, 128, 128))
        stack_target = hvstack(target)
        show_view = np.hstack([stack_frame, stack_events, stack_target])
        cv2.imshow("Dataset Preview", show_view)
        cv2.waitKey(1)

# data_type 0-src_frame, 1-event_voxel, 2-dst_frame
def load_data(args, data_type):
    dataset = DAVIS_Dataset(args.dataset, 0.02, cv2.COLOR_BGR2GRAY, data_type)
    train_loader = data.DataLoader(dataset, batch_size=args.batch_size, shuffle=False)
    return train_loader

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)

def plot_images(images):
    x = images["input"]
    reconstruction = images["rec"]
    half_sample = images["half_sample"]
    full_sample = images["full_sample"]

    fig, axarr = plt.subplots(1, 4)
    axarr[0].imshow(x.cpu().detach().numpy()[0].transpose(1, 2, 0))
    axarr[1].imshow(reconstruction.cpu().detach().numpy()[0].transpose(1, 2, 0))
    axarr[2].imshow(half_sample.cpu().detach().numpy()[0].transpose(1, 2, 0))
    axarr[3].imshow(full_sample.cpu().detach().numpy()[0].transpose(1, 2, 0))
    plt.show()

# preview DAVIS dataset
if __name__ == '__main__':
    dataset = DAVIS_Dataset("Indoor4", 0.02, cv2.IMREAD_GRAYSCALE)
    while(True):
        dataset.__show__(int(input("Index to preview: ")))