import tkinter as tk
from tkinter import messagebox
from ruamel.yaml import YAML
import os


class App:
    def __init__(self, root):
        self.root = root
        self.yaml = YAML()

        # Create the input fields
        self.model_field = self.create_input_field("Model", 0)
        self.model_args_fields = self.create_model_args_fields(1)
        self.tasks_field = self.create_input_field("Tasks", 6)
        self.num_fewshot_field = self.create_input_field("Num Fewshot", 7)
        self.batch_size_field = self.create_input_field("Batch Size", 8)
        self.device_field = self.create_input_field("Device", 9)
        self.output_path_field = self.create_input_field("Output Path", 10)
        # Continue this for other fields

        self.save_button = tk.Button(root, text='Save Configuration', command=self.save_config)
        self.save_button.grid(row=20, column=1)

    def create_input_field(self, label, row):
        label = tk.Label(self.root, text=label)
        label.grid(row=row, column=0)

        field = tk.Entry(self.root)
        field.grid(row=row, column=1)

        return field

    def create_model_args_fields(self, start_row):
        args = ["pretrained", "trust_remote_code", "use_accelerate", "temperature"]
        fields = {}
        for i, arg in enumerate(args):
            row = start_row + i
            label = tk.Label(self.root, text=arg)
            label.grid(row=row, column=0)

            field = tk.Entry(self.root)
            field.grid(row=row, column=1)
            fields[arg] = field

        return fields

    def save_config(self):
        model_args = ",".join([f"{arg}={field.get()}" for arg, field in self.model_args_fields.items()])

        config = {
            'model': self.model_field.get(),
            'model_args': model_args,
            'tasks': self.tasks_field.get(),
            'num_fewshot': self.num_fewshot_field.get(),
            'batch_size': self.batch_size_field.get(),
            'device': self.device_field.get(),
            'output_path': self.output_path_field.get(),
            # Add all other fields
        }

        if not os.path.exists('config.yaml'):
            with open('config.yaml', 'w') as f:
                self.yaml.dump(config, f)
            messagebox.showinfo("Saved", "Configuration saved successfully!")
        else:
            messagebox.showerror("File exists", "config.yaml already exists.")


if __name__ == '__main__':
    root = tk.Tk()
    app = App(root)
    root.mainloop()
