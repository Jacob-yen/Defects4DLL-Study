# def check whether environments exist
import os
import sys
import pandas as pd
import subprocess


CONDA_ROOT = "/root/anaconda3/envs"

# get the parent file directory of the current file
file_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
def get_bug_list(cuda_version, framework):
    assert framework in ["pytorch", "tensorflow"], "Framework must be either 'pytorch' or 'tensorflow'."
    framework_name = "torch" if framework == "pytorch" else "tf"
    framework_bug_file = os.path.join(file_path, f"{framework}-release.xlsx")
    # load the file into a pandas DataFrame
    df = pd.read_excel(framework_bug_file, engine='openpyxl')
    # load the bug list from column "pr_id"
    bugs = df["pr_id"].tolist()
    if framework == "tensorflow":
        assert cuda_version in ["cuda11"], "CUDA version must be '11' for TensorFlow."
    else:
        assert cuda_version in ["cuda10", "cuda11"], "CUDA version must be '10' or '11' for PyTorch."
        # get the bugs for the specified cuda version
        if cuda_version == "cuda11":
            bugs = [b for b in bugs if int(b) >= 62257 ]
        else:
            bugs = [b for b in bugs if int(b) < 62257 ]

    return bugs


def check_execution_environment(env_name,framework):
    # using the installed environment to run a simple command
    print(f"Checking execution environment {env_name}...")
    # get script path
    script_path = os.path.join(file_path, framework, "Result", env_name.replace("-buggy", ""), env_name.replace("-buggy", "-original.py"))
    if not os.path.exists(script_path):
        raise FileNotFoundError(f"Script {script_path} does not exist.")
    else:
        # activate the environment and run the script
        try:
            result = subprocess.run(
                ["conda", "run", "-n", env_name, "python", script_path],
                check=True,
                capture_output=True,
                text=True
            )
            print("The bug is not successfully reproduced!. Plz check the environment.")
            print(result.stdout)
        except subprocess.CalledProcessError as e:
            print("The bug is reproduced!:")
            print("##### Triggered exception:")
            stderr_msg = e.stderr.strip()
            # get the last 10 lines of the stderr message
            last_10_lines = "\n".join(stderr_msg.splitlines()[-20:])
            print(last_10_lines)
            # print the saved traced
            trace_path = os.path.join(file_path, framework, "Result", env_name.replace("-buggy", ""), "stack_trace.txt")
            if os.path.exists(trace_path):
                with open(trace_path, "r") as f:
                    print("##### Saved trace:")
                    saved_trace = f.read()
                    # get the last 10 lines of the saved trace
                    last_10_saved_trace = "\n".join(saved_trace.splitlines()[-20:])
                    print(last_10_saved_trace)




if __name__ == "__main__":
    cuda_version = sys.argv[1] if len(sys.argv) > 1 else "cuda11"
    framework = sys.argv[2] if len(sys.argv) > 2 else "tensorflow"
    bug_list = get_bug_list(cuda_version=cuda_version, framework=framework)
    framework_name = "torch" if framework == "pytorch" else "tf"
    for bug_id in bug_list:
        env_name = f"{framework_name}-{str(bug_id)}-buggy"
        check_execution_environment(env_name, framework)


