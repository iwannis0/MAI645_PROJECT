import read_bvh
import numpy as np
from os import listdir
import os


standard_bvh_file = "train_data_bvh/standard.bvh"
weight_translation = 0.01
skeleton, non_end_bones = read_bvh.read_bvh_hierarchy.read_bvh_hierarchy(standard_bvh_file)


def add_slash(path):
    if path.endswith("/"):
        return path
    return path + "/"


def generate_euler_traindata_from_bvh(src_bvh_folder, tar_traindata_folder):
    src_bvh_folder = add_slash(src_bvh_folder)
    tar_traindata_folder = add_slash(tar_traindata_folder)

    if not os.path.exists(tar_traindata_folder):
        os.makedirs(tar_traindata_folder)

    bvh_dances_names = listdir(src_bvh_folder)

    for bvh_dance_name in bvh_dances_names:
        if bvh_dance_name.endswith(".bvh"):
            print("Encoding Euler:", bvh_dance_name)

            raw_data = read_bvh.parse_frames(src_bvh_folder + bvh_dance_name)

            euler_data = np.array(raw_data, dtype=np.float32)

            # Scale only the hip/global translation.
            # Rotations remain in degrees.
            euler_data[:, 0:3] = euler_data[:, 0:3] * weight_translation

            np.save(tar_traindata_folder + bvh_dance_name + ".npy", euler_data)


def generate_bvh_from_euler_traindata(src_train_folder, tar_bvh_folder):
    src_train_folder = add_slash(src_train_folder)
    tar_bvh_folder = add_slash(tar_bvh_folder)

    if not os.path.exists(tar_bvh_folder):
        os.makedirs(tar_bvh_folder)

    dances_names = listdir(src_train_folder)

    for dance_name in dances_names:
        if dance_name.endswith(".npy"):
            print("Decoding Euler:", dance_name)

            euler_data = np.load(src_train_folder + dance_name)

            bvh_data = np.array(euler_data, dtype=np.float32)

            # Undo translation scaling before writing BVH.
            bvh_data[:, 0:3] = bvh_data[:, 0:3] / weight_translation

            read_bvh.write_frames(
                standard_bvh_file,
                tar_bvh_folder + dance_name + ".bvh",
                bvh_data
            )


bvh_dir_path = "train_data_bvh/martial/"
euler_enc_dir_path = "train_data_euler/martial/"
bvh_reconstructed_dir_path = "reconstructed_bvh_data_euler/martial/"


generate_euler_traindata_from_bvh(bvh_dir_path, euler_enc_dir_path)

generate_bvh_from_euler_traindata(euler_enc_dir_path, bvh_reconstructed_dir_path)