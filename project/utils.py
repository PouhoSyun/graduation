# data fetching and preprocessing module

import mat73, cv2, os, math
import numpy as np
import torch
import torch.nn as nn
import torch.utils.data as data
import matplotlib.pyplot as plt
import albumentations as alb

# 4 basic methods for numpy.array -- fundamental stack/split & fine stack/split functions
# hvstack -- array[n*n, c, size, size] -> array[c, n*size, n*size]
def hvstack(src: np.ndarray, dim4=True):
    if not dim4: 
        for i in src: i = np.array([i])
    dst = []
    for c in src.transpose(1, 0, 2, 3):
        size = int(src.shape[2])
        pieces = int(src.shape[0] ** 0.5)
        lines = []
        for line in src.reshape(pieces, pieces, size, size):
            lines.append(np.hstack(line))
        dst.append(np.vstack(lines).astype(np.uint8))
    if not dim4:
        dst = dst[0]
    return np.array(dst)

# hvsplit -- array[c, n*size, n*size] -> array[n*n, c, size, size]
def hvsplit(src: np.ndarray, pieces = 4, dim4=True):
    if not dim4:
        src = np.array([c])
    dst = []
    for c in src:
        lines = np.vsplit(src, pieces)
        dst_c = []
        for line in lines:
            line = np.hsplit(line, pieces)
            dst_c.extend(line)
        dst.append(dst_c)
    dst = np.array(dst).transpose(1, 0, 2, 3)
    if not dim4:
        for i in dst: i = i[0]
    return dst

# fine stack -- array[n*n, c, size, size] -> array[c, n*size, n*size]
def fine_stack(src: np.ndarray):
    dst = []
    for c in src.transpose(1, 0, 2, 3):
        size = int(c.shape[1])
        pieces = int(c.shape[0] ** 0.5)
        c = hvstack(c, dim4=False)
        dst_c = []
        for i in range(size * size):
            dst_c.append(c[(i//size)::size, (i%size)::size].reshape(pieces, pieces))
        dst.append(dst_c)
    dst = np.array(dst)
    return hvstack(dst)

# fine split -- array[c, n*size, n*size] -> array[n*n, c, size, size]
def fine_split(src: np.ndarray, pieces = 4):
    dst = []
    for c in src:
        size = int(c.shape[0] / pieces)
        dst_c = []
        for i in range(pieces * pieces):
            dst_c.append(c[(i//pieces)::pieces, (i%pieces)::pieces].reshape(size, size))
        dst.append(dst_c)
    dst = np.array(dst).transpose(1, 0, 2, 3)
    return dst

# size of event camera is 190*180
# transform matfile to streamfile, then use spatial-temporal voxel grid method to record events.
def pack_event_stream(dataset, time, split=True):
    try:
        event_countmap = np.load("dataset_prep/" + dataset + "/events_voxel.npy")
    except:
        file_cnt = len(os.listdir("test/dataset/" + dataset + "/events_clip"))
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
    print("Loading image " + str(file_id+1))
    frame = cv2.imread("test/dataset/" + dataset + "/RGB_frame/frame" + str(file_id+1) + ".png", cmap)
    h_pad = (1600 - frame.shape[1]) // 2
    v_pad = (1600 - frame.shape[0]) // 2 
    frame = np.array([cv2.copyMakeBorder(frame, v_pad, v_pad, h_pad, h_pad, cv2.BORDER_CONSTANT, value=(128, 128, 128))])
    if not split: return frame
    else: return fine_split(frame).astype(np.uint8)

def pack_frame_png(dataset, cmap):
    file_cnt = len(os.listdir("test/dataset/" + dataset + "/RGB_frame"))
    frames = []
    for iter in range(file_cnt):
        frame = load_frame_png(dataset, iter, cmap)
        frames.append(frame)
    return np.array(frames)

# get frame dataset: cmap--cv2.IMREAD_*, size--square edge length of the image
class Frame_Dataset(data.Dataset):
    def __init__(self, dataset, cmap, size):
        self.images = pack_frame_png(dataset, cmap)
        self._length = len(self.images)
        self.cmap = cmap
        self.dataset = dataset
        
        self.rescaler = alb.SmallestMaxSize(max_size=size)
        self.cropper = alb.CenterCrop(height=size, width=size)
        self.preprocessor = alb.Compose([self.rescaler, self.cropper])
        pass

    def __len__(self):
        return self._length * 16
    
    def __getitem__(self, index):
        item = self.images[index//16]
        item = self.preprocessor(image=item[index%16].transpose(1, 2, 0))["image"]
        item = (item / 127.5 - 1.0).astype(np.float32)
        return torch.Tensor(item.transpose(2, 0, 1))

#dataset class for dataloader
class DAVIS_Dataset(data.Dataset):
    def __init__(self, dataset, time):
        self.data = pack_event_stream(dataset, time)
        pass

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        return self.data[index]
    
    # def __show__(self, index):
    #     frame, events, target = self.__getitem__(index)
    #     cv2.namedWindow("Dataset Preview", cv2.WINDOW_FREERATIO)
    #     stack_frame = hvstack(frame)
    #     stack_events = cv2.copyMakeBorder(hvstack(events), 700, 700, 0, 0, cv2.BORDER_CONSTANT, value=(128, 128, 128))
    #     stack_target = hvstack(target)
    #     show_view = np.hstack([stack_frame, stack_events, stack_target])
    #     cv2.imshow("Dataset Preview", show_view)
    #     cv2.waitKey(1)

# data_type 0-src_frame, 1-event_voxel, 2-dst_frame
def load_frameset(args):
    dataset = Frame_Dataset(args.dataset, cv2.IMREAD_GRAYSCALE, args.image_size)
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