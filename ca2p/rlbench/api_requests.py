import requests
import time
import paramiko
from scp import SCPClient

BASE_URL = "YOUR_SERVER_ADDRESS_HERE"  # Replace with your FastAPI server address

def init_task(task_name, perturbation: bool = False, adaptation: bool = False):
    print("Adaptation: ", adaptation)
    try:
        # 1. Initialize task session
        init_resp = requests.post(
            f"{BASE_URL}/init_task",
            json={"task": task_name, "perturbation": perturbation, "adaptation": adaptation}
        )
        init_data = init_resp.json()
        session_id = init_data["session_id"]
        initial_objects = init_data["objects"]
        is_success = init_data["is_success"]
        initial_instruction = init_data["initial_instruction"]
        descriptions = init_data["descriptions"]
        if not is_success:
            return False, 0, "", "", ""
        print("Initial objects:", initial_objects)

        if len(initial_objects) > 0:
            req_success = True
        else:
            req_success = False
        return req_success, session_id, initial_objects, initial_instruction, descriptions
    except:
        return False, 0, "", "", ""

def run_task(session_id):
    # 2. Step through the loop until 'done' (success) is returned
    is_terminated = False

    try:
        run_resp = requests.post(
            f"{BASE_URL}/run",
            json={"session_id": session_id, "command": "run"}
        )
        run_data = run_resp.json()
        print("Step result:", run_data)

        # Check for completion flag
        if run_data.get("success"):
            is_terminated = True
            print("Success message received.")
        
        if run_data.get("terminated"):
            is_terminated = True
            print("Terminated with PathOutofError.")
        
        return is_terminated, session_id, run_data
    except Exception as e:
        return False, session_id, None

def shutdown_task(session_id):
    # 3. Shutdown the loop/session
    shutdown_resp = requests.post(
        f"{BASE_URL}/shutdown",
        json={"session_id": session_id}
    )
    print("Shutdown response:", shutdown_resp.json())

class SSHClient:
    def __init__(self):
        import config_remote
        self.config = config_remote

    def create_ssh_client(self) -> bool:
        try:
            self.ssh_client = paramiko.SSHClient()
            self.ssh_client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
            self.ssh_client.connect(
                self.config.REMOTE_HOST,
                username=self.config.REMOTE_USER,
                password=self.config.REMOTE_PASSWORD,
                timeout=10
            )
            print(f"Successfully connected to remote server: {self.config.REMOTE_HOST}")
            return True
        except Exception as e:
            print(f"Failed to connect to remote server: {e}")
            return False

    def send_file(self, local_path: str, remote_path: str):
        if not self.ssh_client:
            print("SSH client not initialized.")
            return False
        try:
            with SCPClient(self.ssh_client.get_transport()) as scp:
                scp.put(local_path, remote_path)
            print(f"File {local_path} sent to remote server at {remote_path}")
            return True
        except Exception as e:
            print(f"Failed to send file to remote server: {e}")
            return False
    
    def close(self):
        self.ssh_client.close()
