# CA^2P Execution Guide

## Alfred Teach Experiments
- Start the simulator using the HELPER paper setup on the simulation server.
- Run the CA^2P pipeline locally with:
  ```bash
  python catp_alf_teach.py
  ```

## RLBench Experiments
- Launch the RLBench simulator on the simulation server first.
- Execute the prompt-only CA^2P driver from your client machine, for example:
  ```bash
  python catp_prompt.py --output ./output.txt --modelname Qwen/Qwen2.5-Coder-14B-Instruct --randomSeed 0 --maxQuestion 1 > ./logs/output.out
  ```
  Adjust arguments as needed for different models, seeds, or question counts.

## Real-World Experiments
- Run the real-world control script (e.g., `combined_methods_realworld.py`) to generate the task code.
- Transfer the generated script to the robot control machine and execute it using the lab’s deployment procedure.
