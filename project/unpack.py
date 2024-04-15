import scipy
import numpy as np

def unpack(dataset_mat):
    path = "dataset_davis/"
    data = scipy.io.loadmat(path + dataset_mat + ".mat")
    data = data['aedat']['data'][0][0][0][0]
    x = data[0][0]['x'][0]
    y = data[0][0]['y'][0]
    t = data[0][0]['timeStamp'][0]
    p = data[0][0]['polarity'][0]
    frames = data[1][0][0]['samples']
    frames = np.array([frame.tolist() for frame in frames])

    events = [[]]
    flag = 1
    frame_times = (data[1][0][0]['timeStampStart'] + data[1][0][0]['timeStampEnd']) / 2
    for event in tuple(zip(x, y, p, t)):
        event = (event[0][0], event[1][0], event[2][0], event[3][0])
        if(event[3] <= frame_times[flag]):
            events[-1].append(event)
        else:
            events[-1] = np.array(events[-1])
            events.append([event])
            flag += 1
            if(flag == len(frame_times)): break
    events[-1] = np.array(events[-1])
    return events, frames
    
if __name__ == '__main__':
    events, frames = unpack("1")
    pass