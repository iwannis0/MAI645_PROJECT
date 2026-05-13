import os
import random
import numpy as np
import torch
import torch.nn as nn


class acLSTM(nn.Module):
    def __init__(self, in_frame_size, hidden_size=1024, out_frame_size=None):
        super(acLSTM, self).__init__()

        if out_frame_size is None:
            out_frame_size = in_frame_size

        self.in_frame_size = in_frame_size
        self.hidden_size = hidden_size
        self.out_frame_size = out_frame_size

        self.lstm1 = nn.LSTMCell(self.in_frame_size, self.hidden_size)
        self.lstm2 = nn.LSTMCell(self.hidden_size, self.hidden_size)
        self.lstm3 = nn.LSTMCell(self.hidden_size, self.hidden_size)
        self.decoder = nn.Linear(self.hidden_size, self.out_frame_size)

    def init_hidden(self, batch, device):
        c0 = torch.zeros(batch, self.hidden_size, device=device)
        c1 = torch.zeros(batch, self.hidden_size, device=device)
        c2 = torch.zeros(batch, self.hidden_size, device=device)

        h0 = torch.zeros(batch, self.hidden_size, device=device)
        h1 = torch.zeros(batch, self.hidden_size, device=device)
        h2 = torch.zeros(batch, self.hidden_size, device=device)

        return [h0, h1, h2], [c0, c1, c2]

    def forward_lstm(self, in_frame, vec_h, vec_c):
        vec_h0, vec_c0 = self.lstm1(in_frame, (vec_h[0], vec_c[0]))
        vec_h1, vec_c1 = self.lstm2(vec_h[0], (vec_h[1], vec_c[1]))
        vec_h2, vec_c2 = self.lstm3(vec_h[1], (vec_h[2], vec_c[2]))

        out_frame = self.decoder(vec_h2)

        vec_h_new = [vec_h0, vec_h1, vec_h2]
        vec_c_new = [vec_c0, vec_c1, vec_c2]

        return out_frame, vec_h_new, vec_c_new

    def forward(self, initial_seq, generate_frames_number):
        device = initial_seq.device
        batch = initial_seq.size(0)

        vec_h, vec_c = self.init_hidden(batch, device)

        outputs = []

        out_frame = torch.zeros(batch, self.out_frame_size, device=device)

        for i in range(initial_seq.size(1)):
            in_frame = initial_seq[:, i]
            out_frame, vec_h, vec_c = self.forward_lstm(in_frame, vec_h, vec_c)

        for i in range(generate_frames_number):
            in_frame = out_frame
            out_frame, vec_h, vec_c = self.forward_lstm(in_frame, vec_h, vec_c)
            outputs.append(out_frame.unsqueeze(1))

        return torch.cat(outputs, dim=1)


def load_dances(dances_folder):
    dances = []

    for filename in os.listdir(dances_folder):
        if filename.endswith(".npy"):
            path = os.path.join(dances_folder, filename)
            dance = np.load(path)
            dances.append(dance)

    return dances


def evaluate_representation(name, dances_folder, weight_path, frame_size,
                            initial_frames=20, predict_frames=20,
                            samples_per_dance=5, seed=42):
    random.seed(seed)
    np.random.seed(seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dances = load_dances(dances_folder)

    model = acLSTM(frame_size, hidden_size=1024, out_frame_size=frame_size)
    model.load_state_dict(torch.load(weight_path, map_location=device))
    model.to(device)
    model.eval()

    mse_values = []

    with torch.no_grad():
        for dance in dances:
            required_length = initial_frames + predict_frames + 20

            if dance.shape[0] < required_length:
                continue

            max_start = dance.shape[0] - initial_frames - predict_frames - 1

            for _ in range(samples_per_dance):
                start = random.randint(0, max_start)

                initial_seq_np = dance[start:start + initial_frames]
                real_future_np = dance[start + initial_frames:
                                      start + initial_frames + predict_frames]

                initial_seq = torch.tensor(
                    initial_seq_np,
                    dtype=torch.float32,
                    device=device
                ).unsqueeze(0)

                real_future = torch.tensor(
                    real_future_np,
                    dtype=torch.float32,
                    device=device
                ).unsqueeze(0)

                predicted_future = model.forward(initial_seq, predict_frames)

                mse = torch.mean((predicted_future - real_future) ** 2)
                mse_values.append(mse.item())

    mean_mse = float(np.mean(mse_values))
    std_mse = float(np.std(mse_values))

    return {
        "Representation": name,
        "Frame size": frame_size,
        "Mean MSE": mean_mse,
        "Std MSE": std_mse,
        "Samples": len(mse_values)
    }


def main():
    experiments = [
        {
            "name": "Positional",
            "dances_folder": "train_data_pos/martial/",
            "weight_path": "weights/pos/0000000.weight",
            "frame_size": 171
        },
        {
            "name": "Euler",
            "dances_folder": "train_data_euler/martial/",
            "weight_path": "weights/euler/0000000.weight",
            "frame_size": 132
        },
        {
            "name": "Quaternion",
            "dances_folder": "train_data_quad/martial/",
            "weight_path": "weights/quad/0000000.weight",
            "frame_size": 175
        }
    ]

    results = []

    for experiment in experiments:
        print("Evaluating:", experiment["name"])

        result = evaluate_representation(
            name=experiment["name"],
            dances_folder=experiment["dances_folder"],
            weight_path=experiment["weight_path"],
            frame_size=experiment["frame_size"],
            initial_frames=20,
            predict_frames=20,
            samples_per_dance=5
        )

        results.append(result)

    print("\nEvaluation results")
    print("-" * 80)
    print(f"{'Representation':<15} {'Frame size':<12} {'Mean MSE':<15} {'Std MSE':<15} {'Samples':<10}")
    print("-" * 80)

    for result in results:
        print(
            f"{result['Representation']:<15} "
            f"{result['Frame size']:<12} "
            f"{result['Mean MSE']:<15.6f} "
            f"{result['Std MSE']:<15.6f} "
            f"{result['Samples']:<10}"
        )

    print("-" * 80)

    np.save("outputs/evaluation_mse_results.npy", results)
    print("Saved results to outputs/evaluation_mse_results.npy")


if __name__ == "__main__":
    os.makedirs("outputs", exist_ok=True)
    main()
