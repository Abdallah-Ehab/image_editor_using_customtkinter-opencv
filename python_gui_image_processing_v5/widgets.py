import tkinter.ttk as ttk

import customtkinter as ctk

class ImageImport(ctk.CTkFrame):
    def __init__(self,master,import_function):
        super().__init__(master=master)
        self.grid(row=0,column=0,sticky='nsew',columnspan=3)

        ctk.CTkButton(master= self,text='open_image',command=self.open_file).pack(expand=True)

        self.import_function = import_function
    def open_file(self):
        self.file_path = ctk.filedialog.askopenfilename(filetypes=[("Image Files", "*.jpg;*.jpeg;*.png")])
        self.import_function(self.file_path)

class image_layout(ctk.CTkCanvas):
    def __init__(self,master,place_image):
        super().__init__(master=master,bg="#333333",bd=0,highlightthickness=0,relief="ridge")
        self.grid(row=0,column=1,sticky="nesw")

        self.bind("<Configure>",place_image)



class manip_tab(ctk.CTkTabview):
    def __init__(self,master,contrast_var,brightness_var,sobel,prewitt,robert,reset,
                 filters,hough_line,hough_circle,max_distance,min_distance,
                 iterations,kernel_size_erosion,erode,dilate
                 ,dilation_kernel_size,dilation_iterations,reset_filter,apply_filters,apply_contrast,
                 max_line_gap,min_line_distance,threshold,
                 kernel_size_open,kernel_size_close,apply_open,apply_close,
                 threshold_segmentaion,apply_segmentation):
        super().__init__(master=master)
        self.grid(row=0,column=0,sticky="nesw")

        self.add('b and c')
        self.add('edge_detection')
        self.add("filters")
        self.add("hough transform")
        self.add('erosion_and_dilation')
        self.add("open_close")
        self.add("segmentation")

        self.contrast_and_brightness = contrast_and_brightness_frame(master=self.tab("b and c"),contrast_var=contrast_var,brightness_var=brightness_var,apply_contrast=apply_contrast)
        self.edge_detection_tab = edge_detection_frame(master=self.tab('edge_detection'),sobel=sobel,prewitt=prewitt,robert=robert,reset=reset)
        self.blur_frame = blurring_filters_frame(master=self.tab('filters'),lpf_val=filters["lpf"],hpf_val=filters["hpf"],mean_val=filters["mean"],median_val=filters["median"],reset_filter=reset_filter,apply=apply_filters)

        self.hough_transform_frame = hough_transform_frame(master=self.tab("hough transform"),hough_circle=hough_circle,max_distance=max_distance,min_distance=min_distance)
        self.hough_transform_frame_line = hough_transform_frame_line(master=self.tab("hough transform"),hough_line=hough_line,max_line_gap=max_line_gap,min_line_distance=min_line_distance,threshold=threshold)
        self.erosion_and_dilation_frame = erosion_and_dilation_frame(master=self.tab('erosion_and_dilation'),iterations=iterations,kernel_size=kernel_size_erosion,erode=erode,dilate=dilate,dilation_iterations=dilation_iterations,dilation_kernel_size=dilation_kernel_size)
        self.open_close_frame = open_close_frame(master=self.tab("open_close"),kernel_size_open=kernel_size_open,kernel_size_close=kernel_size_close,apply_open=apply_open,apply_close=apply_close)

        self.segmentation_frame = segmentation_frame(master=self.tab("segmentation"),threshold_segmentaion=threshold_segmentaion,apply_segmentation=apply_segmentation)


class contrast_and_brightness_frame(ctk.CTkFrame):
    def __init__(self,master,contrast_var,brightness_var,apply_contrast):
        super().__init__(master=master,fg_color="#424242")
        self.pack(expand=True,fill="both")

        self.contrast_card = slider_card(master=self,text="contrast",data_variable=contrast_var,min=0,max=2)
        self.brightness_card = slider_card(master=self,text='brightness',data_variable=brightness_var,min=0,max=100)

        self.apply_contrast_btn=ctk.CTkButton(master=self,text='apply',command=apply_contrast).pack(pady=10)


class edge_detection_frame(ctk.CTkFrame):
    def __init__(self,master,sobel,robert,prewitt,reset):
        super().__init__(master=master,fg_color='#424242')

        #styling
        self.pack(expand=True, fill="both")

        #buttons to activate the filter
        self.sobel_button = ctk.CTkButton(master=self,text='sobel',command=sobel).pack(pady=10)
        self.robert_button = ctk.CTkButton(master=self,text='robert',command=robert).pack(pady=10)
        self.prewitt_button = ctk.CTkButton(master=self,text='prewitt',command=prewitt).pack(pady=10)

        #reset button to remove the filters:(make a copy of the original cv2 image then
        # 1-transform the original cv2 image to pil image
        # 2-place the pil image
        # 3-resize)
        self.reset_button = ctk.CTkButton(master=self,text='reset',command=reset).pack(pady=10)


class blurring_filters_frame(ctk.CTkFrame):
    def __init__(self,master,lpf_val,hpf_val,mean_val,median_val,reset_filter,apply):
        super().__init__(master=master,fg_color="#424242")

        self.pack(expand=True, fill="both")

        self.lpf = slider_card(master=self,text="low_pass",data_variable=lpf_val,min=1,max=10)
        self.hpf = slider_card(master=self,text="high_pass",data_variable=hpf_val,min=1,max=10)
        self.mean = slider_card(master=self,text="mean",data_variable=mean_val,min=1,max=10)
        self.median = slider_card(master=self,text="median",data_variable=median_val,min=1,max=10)

        self.lpf.slider.bind("<Button-1>", lambda event, filter_name="lpf": reset_filter(filter_name,event))
        self.hpf.slider.bind("<Button-1>", lambda event, filter_name="hpf": reset_filter(filter_name,event))
        self.mean.slider.bind("<Button-1>", lambda event, filter_name="mean": reset_filter(filter_name,event))
        self.median.slider.bind("<Button-1>", lambda event, filter_name="median": reset_filter(filter_name,event))

        self.apply_btn = ctk.CTkButton(master=self,text="apply",command=apply).pack(pady=10)

class hough_transform_frame(ctk.CTkFrame):
    def __init__(self,master,hough_circle,min_distance,max_distance):
        super().__init__(master=master,fg_color="#424242")
        self.pack(expand=True, fill='both')

        self.min_distance = slider_card(master=self,text="min_radius",data_variable=min_distance,min=5,max=10)
        self.max_distance = slider_card(master=self,text='max_radius',data_variable=max_distance,min=50,max=100)

        self.hough_circle_btn = ctk.CTkButton(master=self,text='hough_circle',command=hough_circle).pack()


class hough_transform_frame_line(ctk.CTkFrame):
    def __init__(self,master,hough_line,threshold,min_line_distance,max_line_gap):
        super().__init__(master=master,fg_color="#424242")
        self.pack(expand=True, fill='both')

        self.threshold = slider_card(master=self,text="threshold",data_variable=threshold,min=50,max=200)
        self.minLineLength = slider_card(master=self,text='minLineLength',data_variable=min_line_distance,min=10,max=100)
        self.max_line_gap= slider_card(master=self,text='maxLinegap',data_variable=max_line_gap,min=1,max=20)

        self.hough_line_btn = ctk.CTkButton(master=self,text="hough_line",command=hough_line).pack()

class erosion_and_dilation_frame(ctk.CTkFrame):
    def __init__(self,master,iterations,kernel_size,erode,dilate,dilation_kernel_size,dilation_iterations):
        super().__init__(master=master,fg_color="#424242")

        self.pack(expand=True, fill='both')

        self.iterations = slider_card(master=self,text='erosion_iterations',min=1,max=10,data_variable=iterations).pack()
        self.size = slider_card(master=self,text="kernel_size",min=3,max=9,data_variable=kernel_size).pack()

        self.erosion_btn = ctk.CTkButton(master=self, text="erosion", command=erode).pack(pady=10)


        self.dilation_iterations = slider_card(master=self,text='dilation_iterations',min=1,max=10,data_variable=dilation_iterations).pack()
        self.dilation_size = slider_card(master=self,text="dilation kernel_size",min=3,max=9,data_variable=dilation_kernel_size).pack()


        self.dilation_btn = ctk.CTkButton(master=self,text="dilation",command=dilate).pack(pady=10)

class open_close_frame(ctk.CTkFrame):
    def __init__(self,master,kernel_size_open,kernel_size_close,apply_open,apply_close):
        super().__init__(master=master,fg_color="#424242")

        self.pack(expand=True, fill='both')

        self.open_slider=slider_card(master=self,text="open_kernel",min=3,max=21,data_variable=kernel_size_open).pack()
        self.apply_open_btn = ctk.CTkButton(master=self,text="apply_open",command=apply_open).pack(pady=10)

        self.close_slider=slider_card(master=self,text="close_kernel",min=3,max=21,data_variable=kernel_size_close).pack()
        self.apply_close_btn=ctk.CTkButton(master=self,text="apply_close",command=apply_close).pack(pady=10)

class segmentation_frame(ctk.CTkFrame):
    def __init__(self,master,threshold_segmentaion,apply_segmentation):
        super().__init__(master=master,fg_color="#424242")
        self.pack(expand=True, fill='both')

        self.segmentation_slider = slider_card(master=self,text="threshold",min=0,max=255,data_variable=threshold_segmentaion).pack()
        self.apply_segmentation_btn=ctk.CTkButton(master=self,text="apply_segmentation",command=apply_segmentation).pack(pady=10)


class manip_panel(ctk.CTkFrame):
    def __init__(self,master):
        super().__init__(master=master,fg_color="#333333")
        self.pack(fill="x",pady=4,ipady=10)


class slider_card(manip_panel):
    def __init__(self,master,text,data_variable,min,max):
        super().__init__(master=master)
        self.label = ctk.CTkLabel(master=self,text=text).pack()
        self.value = ctk.CTkLabel(master=self,textvariable=data_variable).pack()
        self.slider = ctk.CTkSlider(master=self,from_=min,to=max,variable=data_variable,progress_color="#6e97da",button_color="#476fc2")

        self.slider.pack()