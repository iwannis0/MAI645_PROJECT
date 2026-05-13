import os
import random
import numpy as np
import torch
import torch.nn as nn
import read_bvh


standard_bvh_file = "train_data_bvh/standard.bvh"
weight_translation = 0.01

Hidden_size = 1024
In_frame_size = 132
Out_frame_size = 132


class acLSTM(nn.Module):
    def __init__(self, in_frame_size=132, hidden_size=1024, out_frame_size=132):
        super(acLSTM, self).__init__()

        self.in_frame_size = in_frame_size
        self.hidden_size = hidden_size
        self.out_frame_size = out_frame_size

        self.lstm1 = nn.LSTMCell(self.in_frame_size, self.hidden_size)
        self.lstm2 = nn.LSTMCell(self.hidden_size, self.hidden_size)
        self.lstm3 = nn.LSTMCell(self.hidden_size, self.hidden_size)
        self.decoder = nn.Linear(self.hidden_size, self.out_frame_size)

    def init_hidden(self, batch):
        c0 = torch.autograd.Variable(torch.FloatTensor(np.zeros((batch, self.hidden_size))).cuda())
        c1 = torch.autograd.Variable(torch.FloatTensor(np.zeros((batch, self.hidden_size))).cuda())
        c2 = torch.autograd.Variable(torch.FloatTensor(np.zeros((batch, self.hidden_size))).cuda())

        h0 = torch.autograd.Variable(torch.FloatTensor(np.zeros((batch, self.hidden_size))).cuda())
        h1 = torch.autograd.Variable(torch.FloatTensor(np.zeros((batch, self.hidden_size))).cuda())
        h2 = torch.autograd.Variable(torch.FloatTensor(np.zeros((batch, self.hidden_size))).cuda())

        return ([h0, h1, h2], [c0, c1, c2])

    def forward_lstm(self, in_frame, vec_h, vec_c):
        vec_h0, vec_c0 = self.lstm1(in_frame, (vec_h[0], vec_c[0]))
        vec_h1, vec_c1 = self.lstm2(vec_h[0], (vec_h[1], vec_c[1]))
        vec_h2, vec_c2 = self.lstm3(vec_h[1], (vec_h[2], vec_c[2]))

        out_frame = self.decoder(vec_h2)

        vec_h_new = [vec_h0, vec_h1, vec_h2]
        vec_c_new = [vec_c0, vec_c1, vec_c2]

        return out_frame, vec_h_new, vec_c_new

    def forward(self, initial_seq, generate_frames_number):
        batch = initial_seq.size()[0]
        vec_h, vec_c = self.init_hidden(batch)

        out_seq = torch.autograd.Variable(torch.FloatTensor(np.zeros((batch, 1))).cuda())
        out_frame = torch.autograd.Variable(torch.FloatTensor(np.zeros((batch, self.out_frame_size))).cuda())

        for i in range(initial_seq.size()[1]):
            in_frame = initial_seq[:, i]
            out_frame, vec_h, vec_c = self.forward_lstm(in_frame, vec_h, vec_c)
            out_seq = torch.cat((out_seq, out_frame), 1)

        for i in range(generate_frames_number):
            in_frame = out_frame
            out_frame, vec_h, vec_c = self.forward_lstm(in_frame, vec_h, vec_c)
            out_seq = torch.cat((out_seq, out_frame), 1)

        return out_seq[:, 1:out_seq.size()[1]]


def load_dances(dance_folder):
    dance_files = os.listdir(dance_folder)
    dances = []

    for dance_file in dance_files:
        if dance_file.endswith(".npy"):
            print("load", dance_file)
            dance = np.load(os.path.join(dance_folder, dance_file))
            print("frame number:", dance.shape[0])
            dances.append(dance)

    return dances


def euler_train_to_bvh_frames(euler_data):
    bvh_data = np.array(euler_data, dtype=np.float32)
    bvh_data[:, 0:3] = bvh_data[:, 0:3] / weight_translation
    return bvh_data


def test():
    read_weight_path = "weights/euler/0000000.weight"
    write_bvh_motion_folder = "generated_bvh/euler_eval/"
    dances_folder = "train_data_euler/martial/"

    batch = 5
    initial_seq_len = 15
    generate_frames_number = 400

    os.makedirs(write_bvh_motion_folder, exist_ok=True)

    dances = load_dances(dances_folder)

    model = acLSTM(In_frame_size, Hidden_size, Out_frame_size)
    model.load_state_dict(torch.load(read_weight_path))
    model.cuda()
    model.eval()

    dance_batch = []

    for b in range(batch):
        dance = random.choice(dances)
        dance_len = dance.shape[0]

        start_id = random.randint(10, int(dance_len - initial_seq_len - 10))

        sample_seq = []

        for i in range(initial_seq_len):
            sample_seq.append(dance[start_id + i])

        dance_batch.append(sample_seq)

    dance_batch_np = np.array(dance_batch)
    initial_seq = torch.autograd.Variable(torch.FloatTensor(dance_batch_np.tolist()).cuda())

    predict_seq = model.forward(initial_seq, generate_frames_number)

    predicted_np = np.array(predict_seq.data.tolist()).reshape(batch, -1, In_frame_size)

    for b in range(batch):
        euler_seq = predicted_np[b]
        bvh_frames = euler_train_to_bvh_frames(euler_seq)

        out_path = os.path.join(write_bvh_motion_folder, "euler_out_%02d.bvh" % b)
        read_bvh.write_frames(standard_bvh_file, out_path, bvh_frames)

        np.save(os.path.join(write_bvh_motion_folder, "euler_out_%02d.npy" % b), euler_seq)

    print("Euler evaluation files saved in:", write_bvh_motion_folder)


if __name__ == "__main__":
    test()
