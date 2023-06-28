import tkinter as tk
import subprocess
import threading
from queue import Queue, Empty

process = None
input_queue = Queue()

def execute_main_script():
    global process
    process = subprocess.Popen(
        ["python", "-u", "main.py"],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        universal_newlines=True,
        stdin=subprocess.PIPE,
        bufsize=1,
        encoding='utf-8'
    )

    while process.poll() is None:  # Keep reading until the process exits
        line = process.stdout.readline()
        if line.strip():
            input_queue.put(line)  # Add the line to the input queue

def read_input_queue():
    while True:
        try:
            line = input_queue.get(block=False)
            prompt_text = extract_prompt_text(line)
            if prompt_text:
                input_label.config(text=prompt_text)
                input_entry.config(state=tk.NORMAL)
                input_entry.delete(0, tk.END)
                input_entry.focus()
        except Empty:
            pass

def extract_prompt_text(line):
    line = line.strip()
    if line.startswith("Input:"):
        return line[len("Input:"):].strip()
    elif line.startswith(">>>"):
        return line[len(">>>"):].strip()
    elif line.startswith("..."):
        return line[len("..."):].strip()
    elif "(" in line and ")" in line:
        start = line.index("(")
        end = line.index(")")
        return line[start+1:end].strip()
    return None

def start_main_script():
    thread = threading.Thread(target=execute_main_script)
    thread.start()
    read_input_queue()

def submit_input():
    user_input = input_entry.get()
    input_entry.delete(0, tk.END)
    input_entry.config(state=tk.DISABLED)
    process.stdin.write(user_input + '\n')
    process.stdin.flush()

root = tk.Tk()
root.title("Herakles Adventure")
root.geometry("600x500")

text = tk.Text(root, height=20, width=40)
text.pack()

input_frame = tk.Frame(root)
input_frame.pack()

input_label = tk.Label(input_frame, text="Input:")
input_label.pack(side=tk.LEFT)

input_entry = tk.Entry(input_frame, width=30, state=tk.DISABLED)
input_entry.pack(side=tk.LEFT)

submit_button = tk.Button(input_frame, text="Submit", command=submit_input)
submit_button.pack(side=tk.LEFT)

button = tk.Button(root, text="Start", command=start_main_script)
button.pack()

root.mainloop()
