# data fetching and preprocessing module

import mat73, cv2, os, math
import numpy as np
import torch
import torch.utils.data as data
import matplotlib.pyplot as plt

# size of event camera is 190*180
# transform matfile to streamfile, then use spatial-temporal voxel grid method to record events.
def pack_event_stream(dataset, file_cnt, time):
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
            event_field[int(event[1])][int(event[0])] += 1 / (1 + math.exp(time - event[2])) #consider pre/past event affect
        for it in np.nditer(event_field):
            it = 255 / (1 + math.exp(-it))

        # split the output into 16 50*50 pieces
        lines = np.vsplit(event_field, 4)
        events = []
        for line in lines:
            line = np.hsplit(line, 4)
            events.extend(line)
        event_countmap.append([cell.tolist() for cell in events])

    return ev_stream, torch.tensor(event_countmap)

# size of frame camera is 1520 * 1440, then split to 16 400*400 cell-pics.
def load_frame_png(file_id, color, dataset):
    frame = cv2.cvtColor(cv2.imread("test/dataset/" + dataset + "/RGB_frame/frame" + str(file_id) + ".png"), color)
    h_pad = int((1600 - frame.shape[1]) / 2)
    v_pad = int((1600 - frame.shape[0]) / 2)
    frame = torch.tensor(cv2.copyMakeBorder(frame, v_pad, v_pad, h_pad, h_pad, cv2.BORDER_CONSTANT, value=(128, 128, 128)))
    lines = np.vsplit(frame, 4)
    frame = []
    for line in lines:
        line = np.hsplit(line, 4)
        frame.extend(line)
    frame = [cell.tolist() for cell in frame]
    return torch.tensor(frame)

def pack_frame_png(dataset, file_cnt, cmap):
    frames = []
    for iter in range(file_cnt):
        frame = load_frame_png(iter + 1, cmap, dataset)
        frames.append(frame.tolist())
    return torch.tensor(frames)

def show_cell_pics(frame):
    cv2.namedWindow("Frame PNG", cv2.WINDOW_AUTOSIZE)
    lines = []
    for line in frame.reshape(4, 4, 400, 400):
        lines.append(np.hstack(line))
    pics = np.vstack(lines).astype(np.uint8)
    cv2.imshow("Frame PNG", pics)
    cv2.waitKey(0)

# concatenate tensors of event and frame
def concat_tensors(dataset, time, cmap):
    iter_size = len(os.listdir("test/dataset/" + dataset + "/RGB_frame"))
    try:
        frame = np.load("test/dataset/" + dataset + "/concat_frame.npy")
    except:
        frame = pack_frame_png(dataset, iter_size, cmap).tolist()
        np.save("test/dataset/" + dataset + "/concat_frame.npy", frame)
    
    try:
        events = np.load("test/dataset/" + dataset + "/concat_events.npy") 
    except:
        __, events = pack_event_stream(dataset, iter_size, time)
        events = events.tolist()
        np.save("test/dataset/" + dataset + "/concat_events.npy", events)
        
    data = list(zip(frame[0:-1], events, frame[1:]))
    return data

def hvstack16(tensor: torch.Tensor):
    size = int(tensor.shape[1])
    lines = []
    for line in tensor.reshape(4, 4, size, size):
        lines.append(np.hstack(line))
    view = np.vstack(lines).astype(np.uint8)
    return view

#dataset class for dataloader
class DAVIS_Dataset(data.Dataset):
    def __init__(self, dataset, time, cmap):
        self.data = concat_tensors(dataset, time, cmap)
        pass

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        item = self.data[index]
        return torch.Tensor(item[0]), torch.Tensor(item[1]), torch.Tensor(item[2])
    
    def __show__(self, index):
        frame, events, target = self.__getitem__(index)
        cv2.namedWindow("Dataset Preview", cv2.WINDOW_FREERATIO)
        stack_frame = hvstack16(frame)
        stack_events = cv2.copyMakeBorder(hvstack16(events), 700, 700, 0, 0, cv2.BORDER_CONSTANT, value=(128, 128, 128))
        stack_target = hvstack16(target)
        show_view = np.hstack([stack_frame, stack_events, stack_target])
        cv2.imshow("Dataset Preview", show_view)
        cv2.waitKey(1)

# preview DAVIS dataset
if __name__ == '__main__':
    dataset = DAVIS_Dataset("Indoor4", 0.02, cv2.COLOR_BGR2GRAY)
    while(True):
        dataset.__show__(int(input("Index to preview: ")))