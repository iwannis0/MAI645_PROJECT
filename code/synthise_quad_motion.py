import os
import random
import numpy as np
import torch
import torch.nn as nn
import read_bvh
from scipy.spatial.transform import Rotation as R


standard_bvh_file = "train_data_bvh/standard.bvh"
weight_translation = 0.01
skeleton, non_end_bones = read_bvh.read_bvh_hierarchy.read_bvh_hierarchy(standard_bvh_file)

Hidden_size = 1024
In_frame_size = 175
Out_frame_size = 175


class acLSTM(nn.Module):
    def __init__(self, in_frame_size=175, hidden_size=1024, out_frame_size=175):
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


def wxyz_to_xyzw(q):
    return np.array([q[1], q[2], q[3], q[0]], dtype=np.float32)


def normalize_quaternion(q):
    q = np.array(q, dtype=np.float32)
    norm = np.linalg.norm(q)

    if norm == 0:
        return np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)

    return q / norm


def quaternion_to_euler_hip(q_wxyz):
    q_wxyz = normalize_quaternion(q_wxyz)
    q_xyzw = wxyz_to_xyzw(q_wxyz)

    rotation = R.from_quat(q_xyzw)
    x_angle, y_angle, z_angle = rotation.as_euler("xyz", degrees=True)

    return z_angle, y_angle, x_angle


def quaternion_to_euler_joint(q_wxyz):
    q_wxyz = normalize_quaternion(q_wxyz)
    q_xyzw = wxyz_to_xyzw(q_wxyz)

    rotation = R.from_quat(q_xyzw)
    y_angle, x_angle, z_angle = rotation.as_euler("yxz", degrees=True)

    return z_angle, x_angle, y_angle


def quad_train_to_bvh_frames(quad_data):
    number_of_frames = quad_data.shape[0]
    bvh_frame_size = 6 + 3 * len(non_end_bones)

    bvh_data = np.zeros((number_of_frames, bvh_frame_size), dtype=np.float32)

    for frame_id in range(number_of_frames):
        quad_frame = quad_data[frame_id]

        bvh_data[frame_id, 0:3] = quad_frame[0:3] / weight_translation

        hip_quat = quad_frame[3:7]
        hip_z, hip_y, hip_x = quaternion_to_euler_hip(hip_quat)

        bvh_data[frame_id, 3] = hip_z
        bvh_data[frame_id, 4] = hip_y
        bvh_data[frame_id, 5] = hip_x

        read_index = 7

        for bone_index in range(len(non_end_bones)):
            raw_index = 6 + bone_index * 3

            joint_quat = quad_frame[read_index:read_index + 4]
            joint_z, joint_x, joint_y = quaternion_to_euler_joint(joint_quat)

            bvh_data[frame_id, raw_index] = joint_z
            bvh_data[frame_id, raw_index + 1] = joint_x
            bvh_data[frame_id, raw_index + 2] = joint_y

            read_index += 4

    return bvh_data


def test():
    read_weight_path = "weights/quad/0000000.weight"
    write_bvh_motion_folder = "generated_bvh/quad_eval/"
    dances_folder = "train_data_quad/martial/"

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
        quad_seq = predicted_np[b]
        bvh_frames = quad_train_to_bvh_frames(quad_seq)

        out_path = os.path.join(write_bvh_motion_folder, "quad_out_%02d.bvh" % b)
        read_bvh.write_frames(standard_bvh_file, out_path, bvh_frames)

        np.save(os.path.join(write_bvh_motion_folder, "quad_out_%02d.npy" % b), quad_seq)

    print("Quaternion evaluation files saved in:", write_bvh_motion_folder)


if __name__ == "__main__":
    test()
