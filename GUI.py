import tkinter as tk
from tkinter import ttk
from tkinter import filedialog
import threading
import sys
import queue
import logging
import GUI_backend as model

class GUI(tk.Tk):
    def __init__(self):
        tk.Tk.__init__(self)
        self.shared_data = {"input_data":tk.StringVar(),
                            "posterior_data":tk.StringVar(),
                            "output_dir":tk.StringVar(),
                            "output_name":tk.StringVar(),
                            "draws":tk.IntVar(value=2000),
                            "tune":tk.IntVar(value=4000),
                            "n_iter":tk.IntVar(value=1),
                            "hierarchical":tk.StringVar()}
        self.noise_method_var = tk.StringVar()
        self.AB_method_var = tk.StringVar()
        self.img_pick = {"AB":tk.StringVar(),
                         "Noise":tk.StringVar(),
                         "Hyperprior":tk.StringVar(),
                         "Noise_prior_gamma":tk.StringVar(),
                         "Noise_prior_ratio":tk.StringVar(),
                         "Noise_prior_Non_centered":tk.StringVar()}
        self.geometry("550x250+700+300")
        self._frame = None
        self.switch_frame(StartWindow)

    def switch_frame(self, frame_class):
        """Destroys current frame and replaces it with a new one."""
        new_frame = frame_class(self)
        if self._frame is not None:
            self._frame.destroy()
        self._frame = new_frame
        self._frame.pack()

class StartWindow(tk.Frame):
    def __init__(self, master):
        tk.Frame.__init__(self,master)
            
        self.new_model_button = tk.Button(self, text='Run model on new data', command=lambda:master.switch_frame(NewModelWindow))
        # self.prev_data_model_button = tk.Button(self, text='Use previous results as prior', command=lambda:master.switch_frame(PrevDataModelWindow))
        # self.create_img_button = tk.Button(self, text='Create images from netcddf', command=lambda:master.switch_frame(CreateImgWindow))
        self.new_model_button.grid(row=3,pady=4)
        # self.prev_data_model_button.grid(row=1,pady=4)
        # self.create_img_button.grid(row=2,pady=4)

class NewModelWindow(tk.Frame):
    def __init__(self,master):
        tk.Frame.__init__(self,master)

        self.input_data_text = tk.Label(self, text = 'Data File:')
        self.input_data_entry = tk.Entry(self, width=30, textvariable=master.shared_data["input_data"])
        self.input_data_button = tk.Button(self, text="Browse...", command=lambda:self.browse_file(self.input_data_entry))
        self.input_data_text.grid(column=1,row=1,padx=2,pady=4, sticky='e')
        self.input_data_entry.grid(column=2,row=1,columnspan=2,padx=2,pady=4)
        self.input_data_button.grid(column=4,row=1,padx=2,pady=4, sticky='w')

        self.check = tk.BooleanVar()
        self.checkbutton = tk.Checkbutton(self, text="Use posterior data", variable=self.check,onvalue=True, offvalue=False,command=self.checkbutton_command)
        self.checkbutton.grid(column=1,row=2,padx=2,pady=4, sticky='e')

        self.posterior_data_text = tk.Label(self, text = 'Posterior Data File:')
        self.posterior_data_text.grid(column=1,row=3,padx=2,pady=4, sticky='e')
        self.posterior_data_entry = tk.Entry(self, width=30, textvariable=master.shared_data["posterior_data"])
        self.posterior_data_entry.grid(column=2,row=3,columnspan=2,padx=2,pady=4)
        self.posterior_data_button = tk.Button(self, text="Browse...", command=lambda:self.browse_file(self.posterior_data_entry))
        self.posterior_data_button.grid(column=4,row=3,padx=2,pady=4, sticky='w')
        self.posterior_data_entry["state"] = "disabled"
        self.posterior_data_button["state"] = "disabled"

        self.output_dir_text = tk.Label(self, text = 'Output Directory:')
        self.output_dir_text.grid(column=1,row=4,padx=2,pady=4, sticky='e')
        self.output_dir_entry = tk.Entry(self, width=30, textvariable=master.shared_data["output_dir"])
        self.output_dir_entry.grid(column=2,row=4,columnspan=2,padx=2,pady=4)
        self.output_dir_button = tk.Button(self, text="Browse...", command=lambda:self.browse_dir(self.output_dir_entry))
        self.output_dir_button.grid(column=4,row=4,padx=2,pady=4, sticky='w')

        self.output_filename_text = tk.Label(self, text='Output Filename (optional):')
        self.output_filename_text.grid(column=1,row=5,padx=2,pady=4, sticky='e')
        self.output_filename_entry = tk.Entry(self, width=30, textvariable=master.shared_data["output_name"])
        self.output_filename_entry.grid(column=2,row=5,columnspan=2,padx=2,pady=4)

        self.select_hierarchical = tk.Radiobutton(self, text = 'Hierarchical', variable=master.shared_data["hierarchical"], value='Hierarchical')
        self.select_hierarchical.grid(column=2,row=6,padx=2,pady=4)
        self.select_non_hierarchical = tk.Radiobutton(self, text = 'Non Hierarchical', variable=master.shared_data["hierarchical"],value='Non Hierarchical')
        self.select_non_hierarchical.grid(column=3,row=6,padx=2,pady=4)
        if not master.shared_data["hierarchical"].get():
            self.select_non_hierarchical.select()

        self.continue_button = tk.Button(self, text='Continue', command=lambda:self.pick_next_frame(master))
        self.continue_button.grid(column=3, row=7,sticky='S')
        self.advanced_button = tk.Button(self, text='Advanced', command=lambda:master.switch_frame(AdvancedSettingsWindow))
        self.advanced_button.grid(column=2, row=7,sticky='S')

    def pick_next_frame(self,master):
        if master.shared_data["hierarchical"].get() == 'Hierarchical':
            master.switch_frame(HierarchicalWindow)
        elif master.shared_data["hierarchical"].get() == 'Non Hierarchical':
            master.switch_frame(NonHierarchicalWindow)

    def browse_file(self, file_entry):
        input_file = filedialog.askopenfilename()
        file_entry.get
        file_entry.delete(0,tk.END)
        file_entry.insert(0, input_file)

    def browse_dir(self, directory_entry):
        directory = filedialog.askdirectory()
        directory_entry.get
        directory_entry.delete(0, tk.END)
        directory_entry.insert(0, directory)

    def checkbutton_command(self):
        if self.check.get():
            self.posterior_data_entry["state"] = "normal"
            self.posterior_data_button["state"] = "normal"
        else:
            self.posterior_data_entry["state"] = "disabled"
            self.posterior_data_button["state"] = "disabled"

class AdvancedSettingsWindow(tk.Frame):
    def __init__(self,master):
        tk.Frame.__init__(self,master)
        self.draw_label = tk.Label(self,text='Draws:').grid(column=0,row=0,padx=2,pady=4)
        self.draw_entry = tk.Entry(self,textvariable=master.shared_data["draws"])
        self.draw_entry.grid(column=1,row=0,padx=2,pady=4)
        self.tune_label = tk.Label(self,text='Tune:').grid(column=0,row=1,padx=2,pady=4)
        self.tune_entry = tk.Entry(self,textvariable=master.shared_data["tune"])
        self.tune_entry.grid(column=1,row=1,padx=2,pady=4)

        self.return_button = tk.Button(self,text='OK', command=lambda:master.switch_frame(NewModelWindow))
        self.return_button.grid(column=1,row=2,padx=2,pady=4)

class HierarchicalWindow(tk.Frame):
    def __init__(self,master):
        tk.Frame.__init__(self,master)
        self.output = f'{master.shared_data["output_dir"].get()}/{master.shared_data["output_name"].get()}.nc'

        self.iter_label = tk.Label(self,text='Number of iterations:').grid(column=0,row=0,padx=2,pady=4)
        self.iter_entry = tk.Entry(self,textvariable=master.shared_data["n_iter"])
        self.iter_entry.grid(column=1,row=0,padx=2,pady=4)

        noise_methods = ["Gamma","NonCentered", "Ratio"]
        master.noise_method_var.set(noise_methods[0])
        self.method_label = tk.Label(self,text='Method for calculating noise:')
        self.method_label.grid(column=0,row=1,padx=2,pady=4)
        self.method_menu = ttk.Combobox(self, textvariable=master.noise_method_var, values=noise_methods)
        self.method_menu.grid(column=1,row=1,padx=2,pady=4)

        self.save_img_button = tk.Button(self,text='Save images', command=lambda:master.switch_frame(ImageSelectWindow))
        self.save_img_button.grid(column=0,row=2,padx=2,pady=4)

        self.run_button = tk.Button(self,text='Run', command=lambda:self.run_thread(master))
        self.run_button.grid(column=1,row=3,padx=2,pady=4)
        self.return_button = tk.Button(self,text='Back', command=lambda:master.switch_frame(NewModelWindow))
        self.return_button.grid(column=0,row=3,padx=2,pady=4)
        # self.out = ThreadSafeText(self)
        # self.out.grid(column=0,row=3,columnspan=2)
        # redir = IORedirector(self.out)
        # sys.stdout = redir
        # sys.stderr = redir
        # logging.basicConfig(stream=sys.stdout,format='%(message)s')

    def run_thread(self,master):
        th = threading.Thread(target=self.run_hierarchical,args=(master,),daemon=True)
        th.start()

    def run_hierarchical(self,master):
        data = model.TestData(master.shared_data["input_data"].get())
        ssm = model.StateSpace(data,noise_method=master.noise_method_var.get())
        if master.shared_data["posterior_data"].get():
            ssm.OneRateHierarchical_reiterate_posterior(self.output, n_iter=master.shared_data["n_iter"].get())
        elif master.shared_data["n_iter"].get()>1:
            ssm.OneRateNonHierarchical(self.output)
            ssm.OneRateNonHierarchical_reiterate_posterior(output_file=self.output,n_iter=master.shared_data["n_iter"].get()-1)
        else:
            ssm.OneRateHierarchical(output_file=self.output,draw=master.shared_data["draws"].get(),tune=master.shared_data["tune"].get())



class NonHierarchicalWindow(tk.Frame):
    def __init__(self,master):
        tk.Frame.__init__(self,master)
        self.output = f'{master.shared_data["output_dir"].get()}/{master.shared_data["output_name"].get()}.nc'
        self.iter_label = tk.Label(self,text='Number of iterations:').grid(column=0,row=0,padx=2,pady=4)
        self.iter_entry = tk.Entry(self,textvariable=master.shared_data["n_iter"])
        self.iter_entry.grid(column=1,row=0,padx=2,pady=4)
        noise_methods = ["Gamma","NonCentered", "Ratio"]
        master.noise_method_var.set(noise_methods[0])
        self.method_label = tk.Label(self,text='Method for calculating noise:')
        self.method_label.grid(column=0,row=1,padx=2,pady=4)
        self.method_menu = ttk.Combobox(self, textvariable=master.noise_method_var, values=noise_methods)
        self.method_menu.grid(column=1,row=1,padx=2,pady=4)
        AB_methods = ["Informed","Wide", "Non Hierarchical"]
        master.AB_method_var.set(AB_methods[0])
        self.AB_method_label = tk.Label(self,text='Method for calculating learning params:')
        self.AB_method_label.grid(column=0,row=2,padx=2,pady=4)
        self.AB_method_menu = ttk.Combobox(self, textvariable=master.AB_method_var, values=AB_methods)
        self.AB_method_menu.grid(column=1,row=2,padx=2,pady=4)
        self.save_img_button = tk.Button(self,text='Save images', command=lambda:master.switch_frame(ImageSelectWindow))
        self.save_img_button.grid(column=0,row=3,padx=2,pady=4)
        self.run_button = tk.Button(self,text='Run', command=lambda:self.run_thread(master))
        self.run_button.grid(column=1,row=4,padx=2,pady=4)
        self.return_button = tk.Button(self,text='Back', command=lambda:master.switch_frame(NewModelWindow))
        self.return_button.grid(column=0,row=4,padx=2,pady=4)
        
    def run_thread(self,master):
        th = threading.Thread(target=self.run_non_hierarchical,args=(master,),daemon=True)
        th.start()

    def run_non_hierarchical(self,master):
        data = model.TestData(master.shared_data["input_data"].get())
        ssm = model.StateSpace(data,AB_method=master.AB_method_var.get(),noise_method=master.noise_method_var.get())
        if master.shared_data["posterior_data"].get():
            ssm.OneRateNonHierarchical_reiterate_posterior(master.shared_data["posterior_data"].get(),self.output,n_iter=master.shared_data["n_iter"].get())
        elif master.shared_data["n_iter"].get()>1:
            ssm.OneRateNonHierarchical(self.output)
            ssm.OneRateNonHierarchical_reiterate_posterior(output_file=self.output,n_iter=master.shared_data["n_iter"].get()-1)
        else:
            ssm.OneRateNonHierarchical(output_file=self.output, draws=master.shared_data["draws"].get(),tune=master.shared_data["tune"].get())


class IORedirector(object):
    '''A general class for redirecting I/O to this Text widget.'''
    def __init__(self,text_area):
        self.text_area = text_area

    def write(self,str):
        self.text_area.insert(tk.END, str)
        
    def flush(self):
        pass

class ThreadSafeText(tk.Text):
    def __init__(self, master, **options):
        tk.Text.__init__(self, master, **options)
        self.queue = queue.Queue()
        self.update_me()

    def write(self, line):
        self.queue.put(line)

    def update_me(self):
        while not self.queue.empty():
            line = self.queue.get_nowait()
            self.insert(tk.END, line)
            self.see(tk.END)
            self.update_idletasks()
        self.after(10, self.update_me)

class ImageSelectWindow(tk.Frame):
    def __init__(self,master):
        tk.Frame.__init__(self,master)

        options = ['None','Trace','Posterior distribution']
        if master.shared_data["hierarchical"].get() == "Non Hierarchical":
            self.AB_label = tk.Label(self,text="Learning parameters A,B:")
            self.AB_menu = ttk.Combobox(self, textvariable=master.img_pick["AB"], values=options)
            
            self.noise_label = tk.Label(self,text="Noise parameters sigma eta, sigma epsilon:")
            self.noise_menu = ttk.Combobox(self, textvariable=master.img_pick["Noise"],values=options)

            self.back_button = tk.Button(self,text='Back', command=lambda:master.switch_frame(NonHierarchicalWindow))

            self.AB_menu.grid(column=1,row=0)
            self.AB_label.grid(column=0,row=0)
            self.noise_menu.grid(column=1,row=1)
            self.noise_label.grid(column=0,row=1)
            self.back_button.grid(column=1,row=2)

        elif master.shared_data["hierarchical"].get() == "Hierarchical":
            self.AB_label = tk.Label(self,text="Learning parameters A,B:")
            self.AB_menu = ttk.Combobox(self, textvariable=master.img_pick["AB"], values=options)
            
            self.noise_label = tk.Label(self,text="Noise parameters sigma eta, sigma epsilon:")
            self.noise_menu = ttk.Combobox(self, textvariable=master.img_pick["Noise"],values=options)

            if master.noise_method_var.get() == "Ratio":
                self.noise_prior_label = tk.Label(self,text="Noise ratio priors p, pmu, psigma:")
                self.noise_prior_menu = ttk.Combobox(self, textvariable=master.img_pick["Noise_prior_ratio"],values=options)

            elif master.noise_method_var.get() == "Gamma":
                self.noise_prior_label = tk.Label(self,text="Noise hyper-priors sigma eta mu, sigma eta sigma, sigma epsilon mu, sigma epsilon sigma:")
                self.noise_prior_menu = ttk.Combobox(self, textvariable=master.img_pick["Noise_prior_gamma"],values=options)

            elif master.noise_method_var.get() == "NonCentered":
                self.noise_prior_label = tk.Label(self,text="Noise ratio priors eta mode, eta var, epsilon mode, epsilon var :")
                self.noise_prior_menu = ttk.Combobox(self, textvariable=master.img_pick["Noise_prior_Non_centered"],values=options)


            self.back_button = tk.Button(self,text='Back', command=lambda:master.switch_frame(NonHierarchicalWindow))

            self.AB_menu.grid(column=1,row=0)
            self.AB_label.grid(column=0,row=0)
            self.noise_menu.grid(column=1,row=1)
            self.noise_label.grid(column=0,row=1)
            self.noise_prior_label.grid(column=0,row=2)
            self.noise_prior_menu.grid(column=1,row=2)
            self.back_button.grid(column=1,row=3)



            

if __name__ == "__main__":
    window = GUI()
    window.mainloop()
