import os, argparse

from fets_challenge import model_outputs_to_disc


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog="FeTS_Challenge_Inference",
        formatter_class=argparse.RawTextHelpFormatter,
        description="Generate and save predictions from trained models.\n\n",
    )

    parser.add_argument(
        "-d",
        "--data_dir",
        type=str,
        required=True,
        help="Path to the directory containing the data on which inference is to be run.",
    )
    parser.add_argument(
        "-m",
        "--models_dir",
        type=str,
        required=True,
        help="Path to the directory containing all the trained models.",
    )
    parser.add_argument(
        "-0", "--out_dir", type=str, required=True, help="Path to the output directory."
    )
    parser.add_argument(
        "-device",
        default="cuda",
        type=str,
        help="Device to perform requested session on 'cpu' or 'cuda'; for cuda, ensure CUDA_VISIBLE_DEVICES env var is set",
        required=True,
    )
    args = parser.parse_args()

    all_submissions = os.listdir(args.models_dir)

    current_idx = 0

    for submission in all_submissions:
        # get expected model path
        current_model = os.path.join(
            args.models_dir, submission, "new_functionality_false", "best_model.pkl"
        )
        if os.path.exists(current_model):
            current_output_dir = os.path.join(args.out_dir, submission)
            print(
                "Started generating predictions for ",
                current_idx,
                "/",
                len(all_submissions),
            )
            model_outputs_to_disc(
                args.data_dir,
                current_output_dir,
                current_model,
                submission + "_pred_seg",
                args.device,
            )
        else:
            print("No model found:", current_model)

    print("Done.")
