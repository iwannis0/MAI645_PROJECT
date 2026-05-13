import read_bvh
import numpy as np
from os import listdir
import os
from scipy.spatial.transform import Rotation as R


standard_bvh_file = "train_data_bvh/standard.bvh"
weight_translation = 0.01
skeleton, non_end_bones = read_bvh.read_bvh_hierarchy.read_bvh_hierarchy(standard_bvh_file)


def add_slash(path):
    if path.endswith("/"):
        return path
    return path + "/"


def xyzw_to_wxyz(q):
    return np.array([q[3], q[0], q[1], q[2]], dtype=np.float32)


def wxyz_to_xyzw(q):
    return np.array([q[1], q[2], q[3], q[0]], dtype=np.float32)


def normalize_quaternion(q):
    q = np.array(q, dtype=np.float32)
    norm = np.linalg.norm(q)

    if norm == 0:
        return np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)

    return q / norm


def euler_to_quaternion_hip(z_angle, y_angle, x_angle):
    rotation = R.from_euler(
        "xyz",
        [x_angle, y_angle, z_angle],
        degrees=True
    )

    return xyzw_to_wxyz(rotation.as_quat())


def euler_to_quaternion_joint(z_angle, x_angle, y_angle):
    rotation = R.from_euler(
        "yxz",
        [y_angle, x_angle, z_angle],
        degrees=True
    )

    return xyzw_to_wxyz(rotation.as_quat())


def quaternion_to_euler_hip(q_wxyz):
    q_wxyz = normalize_quaternion(q_wxyz)
    q_xyzw = wxyz_to_xyzw(q_wxyz)

    rotation = R.from_quat(q_xyzw)

    x_angle, y_angle, z_angle = rotation.as_euler(
        "xyz",
        degrees=True
    )

    return z_angle, y_angle, x_angle


def quaternion_to_euler_joint(q_wxyz):
    q_wxyz = normalize_quaternion(q_wxyz)
    q_xyzw = wxyz_to_xyzw(q_wxyz)

    rotation = R.from_quat(q_xyzw)

    y_angle, x_angle, z_angle = rotation.as_euler(
        "yxz",
        degrees=True
    )

    return z_angle, x_angle, y_angle


def generate_quad_traindata_from_bvh(src_bvh_folder, tar_traindata_folder):
    src_bvh_folder = add_slash(src_bvh_folder)
    tar_traindata_folder = add_slash(tar_traindata_folder)

    if not os.path.exists(tar_traindata_folder):
        os.makedirs(tar_traindata_folder)

    bvh_dances_names = listdir(src_bvh_folder)

    for bvh_dance_name in bvh_dances_names:
        if bvh_dance_name.endswith(".bvh"):
            print("Encoding Quaternion:", bvh_dance_name)

            raw_data = read_bvh.parse_frames(src_bvh_folder + bvh_dance_name)

            number_of_frames = raw_data.shape[0]

            # Quaternion representation:
            # 3 values for hip translation
            # 4 values for hip rotation
            # 4 values for each non-end bone rotation
            quad_frame_size = 3 + 4 * (1 + len(non_end_bones))

            quad_data = np.zeros(
                (number_of_frames, quad_frame_size),
                dtype=np.float32
            )

            for frame_id in range(number_of_frames):
                raw_frame = raw_data[frame_id]

                # Hip/global translation
                quad_data[frame_id, 0:3] = raw_frame[0:3] * weight_translation

                # Hip rotation
                hip_z = raw_frame[3]
                hip_y = raw_frame[4]
                hip_x = raw_frame[5]

                hip_quat = euler_to_quaternion_hip(
                    hip_z,
                    hip_y,
                    hip_x
                )

                quad_data[frame_id, 3:7] = hip_quat

                # Other joint rotations
                write_index = 7

                for bone_index in range(len(non_end_bones)):
                    raw_index = 6 + bone_index * 3

                    joint_z = raw_frame[raw_index]
                    joint_x = raw_frame[raw_index + 1]
                    joint_y = raw_frame[raw_index + 2]

                    joint_quat = euler_to_quaternion_joint(
                        joint_z,
                        joint_x,
                        joint_y
                    )

                    quad_data[frame_id, write_index:write_index + 4] = joint_quat

                    write_index = write_index + 4

            np.save(tar_traindata_folder + bvh_dance_name + ".npy", quad_data)


def generate_bvh_from_quad_traindata(src_train_folder, tar_bvh_folder):
    src_train_folder = add_slash(src_train_folder)
    tar_bvh_folder = add_slash(tar_bvh_folder)

    if not os.path.exists(tar_bvh_folder):
        os.makedirs(tar_bvh_folder)

    dances_names = listdir(src_train_folder)

    for dance_name in dances_names:
        if dance_name.endswith(".npy"):
            print("Decoding Quaternion:", dance_name)

            quad_data = np.load(src_train_folder + dance_name)

            number_of_frames = quad_data.shape[0]

            # Original BVH raw frame format:
            # 3 hip translation values
            # 3 hip rotation values
            # 3 rotation values for each non-end bone
            bvh_frame_size = 6 + 3 * len(non_end_bones)

            bvh_data = np.zeros(
                (number_of_frames, bvh_frame_size),
                dtype=np.float32
            )

            for frame_id in range(number_of_frames):
                quad_frame = quad_data[frame_id]

                # Hip/global translation
                bvh_data[frame_id, 0:3] = quad_frame[0:3] / weight_translation

                # Hip rotation
                hip_quat = quad_frame[3:7]

                hip_z, hip_y, hip_x = quaternion_to_euler_hip(hip_quat)

                bvh_data[frame_id, 3] = hip_z
                bvh_data[frame_id, 4] = hip_y
                bvh_data[frame_id, 5] = hip_x

                # Other joint rotations
                read_index = 7

                for bone_index in range(len(non_end_bones)):
                    raw_index = 6 + bone_index * 3

                    joint_quat = quad_frame[read_index:read_index + 4]

                    joint_z, joint_x, joint_y = quaternion_to_euler_joint(joint_quat)

                    bvh_data[frame_id, raw_index] = joint_z
                    bvh_data[frame_id, raw_index + 1] = joint_x
                    bvh_data[frame_id, raw_index + 2] = joint_y

                    read_index = read_index + 4

            read_bvh.write_frames(
                standard_bvh_file,
                tar_bvh_folder + dance_name + ".bvh",
                bvh_data
            )


bvh_dir_path = "train_data_bvh/martial/"
quad_enc_dir_path = "train_data_quad/martial/"
bvh_reconstructed_dir_path = "reconstructed_bvh_data_quad/martial/"


generate_quad_traindata_from_bvh(bvh_dir_path, quad_enc_dir_path)

generate_bvh_from_quad_traindata(quad_enc_dir_path, bvh_reconstructed_dir_path)