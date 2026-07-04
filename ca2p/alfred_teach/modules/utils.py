from termcolor import cprint

def info_print(text):
    cprint("[INFO]","blue",end="")
    cprint(f"   {text}","blue")

def debug_print(text):
    cprint("[DEBUG]","yellow",end="")
    cprint(f"   {text}","yellow")

def error_print(text):
    cprint("[ERROR]","red",end="")
    cprint(f"   {text}","red")

def start_print():
    cprint(f"----------CODE START----------","red")

def success_print(text):
    cprint("[SUCCESS]","green",end="")
    cprint(f"   {text}","green")