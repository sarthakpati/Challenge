import os, argparse

from fets_challenge import model_outputs_to_disc




if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog="FeTS_Challenge_Inference",
        formatter_class=argparse.RawTextHelpFormatter,
        description="Generate and save predictions from trained models.\n\n"
    )

    parser.add_argument( "-d", "--data_dir", type=str, required=True, help="Path to the directory containing the data on which inference is to be run." )
    parser.add_argument( "-m", "--models_dir", type=str, required=True, help="Path to the directory containing all the trained models." )
    
    parser.add_argument(
        "-device",
        default="cuda",
        type=str,
        help="Device to perform requested session on 'cpu' or 'cuda'; for cuda, ensure CUDA_VISIBLE_DEVICES env var is set",
        required=True,
    )
    args = parser.parse_args()
