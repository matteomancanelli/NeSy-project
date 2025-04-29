import os
import signal
import pathlib
import shutil
from subprocess import Popen

def extract_plan(trace_text, max_length=None):
    """
    Extract a plan from trace aligner output, keeping only 'add' and 'sync' actions
    while discarding 'del' actions.
    
    Args:
        trace_text (str): The input trace text containing the plan
        
    Returns:
        list: A list of actions in order of execution
    """
    plan = []
    
    # Split the text into lines and process each line
    for line in trace_text.strip().split('\n'):
        # Skip empty lines, comments, and cost information
        if not line or line.startswith(';'):
            continue
            
        # Remove parentheses and split into components
        parts = line.strip('()').split()
        
        if not parts:
            continue
            
        action_type = parts[0]
        
        if action_type == 'add':
            # For 'add' actions, take the action name
            plan.append(parts[1])
        elif action_type == 'sync':
            action = parts[2]
            if action != 'dummy':  # Skip dummy actions
                plan.append(action)
    
    while len(plan) < max_length + 1:
        plan.append("end")

    return plan

def process_trace_file(file_path, max_length=None):
    """
    Process a trace file and extract the plan.
    
    Args:
        file_path (str): Path to the trace file
        
    Returns:
        list: A list of actions in order of execution
    """
    try:
        with open(file_path, 'r') as f:
            trace_text = f.read()
        return extract_plan(trace_text, max_length)
    except FileNotFoundError:
        print(f"Error: File {file_path} not found")
        return []
    except Exception as e:
        print(f"Error processing file: {str(e)}")
        return []

def launch(cmd, cwd=None):
    """Launch a command."""
    print("Running command: ", " ".join(map(str, cmd)))
    process = Popen(args=cmd, encoding="utf-8", cwd=cwd,)
    try:
        process.wait()
    except KeyboardInterrupt:
        print("Interrupted")
    finally:
        if process.poll() is None:
            try:
                print("do killpg")
                os.killpg(os.getpgid(process.pid), signal.SIGTERM)
            except:
                print("killpg failed")
        if process.returncode != 0:
            print(f"return code {process.returncode}")
            exit(1)

def align_traces(log_file, form_file, traces_file, max_length=None):
    base_dir = str(os.getcwd()) + "/"
    curr_dir = base_dir + "trace_alignment/"
    pddl_dir = base_dir + "pddl/"

    pddl_encoding = "3"
    jar_file = curr_dir + "generate_pddl.jar"
    domain_file = pddl_dir + f"domain-e{pddl_encoding}.pddl"
    log_file = base_dir + str(log_file)
    form_file = base_dir + str(form_file)
    plan_file = curr_dir + "sas_plan"

    with open(pathlib.Path(base_dir, traces_file), "w") as out_file:
        #cli_args = ["java", "-Xmx4g", "-jar", jar_file, "--log", log_file, "--formulas", form_file, "--output", pddl_dir, "--encoding", pddl_encoding]
        cli_args = [
            "java", "-Xmx4g",
            "-cp", f"{jar_file}:trace_alignment/picocli-4.7.5.jar",
            "trace_alignment.App",
            "--log", log_file,
            "--formulas", form_file,
            "--output", pddl_dir,
            "--encoding", pddl_encoding
        ]
        launch(cli_args, cwd=curr_dir)

        for problem_file in pathlib.Path(pddl_dir).iterdir():
            if not str(problem_file).split("/")[-1].startswith("p-"):
                continue
            
            cli_args = ["python3", "../submodules/downward/fast-downward.py", domain_file, str(problem_file), "--search", "astar(blind())"]
            launch(cli_args, cwd=curr_dir)

            aligned_trace = process_trace_file(plan_file, max_length)
            out_file.write(f"{aligned_trace}\n")
    
    shutil.rmtree(pddl_dir)
    os.remove(plan_file)
