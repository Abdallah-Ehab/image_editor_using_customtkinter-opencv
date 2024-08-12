import os

import customtkinter as ctk
from widgets import *
import cv2
from PIL import Image,ImageTk
import numpy as np
class App(ctk.CTk):
    def __init__(self):
        super().__init__()
        ctk.set_appearance_mode('dark')
        self.geometry('1000x650')
        self.iconbitmap('icons/lisa_icon.ico')
        self.title('image_processing')

        #layout
        self.rowconfigure(0,weight=1)
        self.columnconfigure(0,weight=3,uniform='a')
        self.columnconfigure(1,weight=6,uniform='a')






        # ctk_variables_global:

        self.contrast = ctk.DoubleVar(value=1)
        self.old_contrast = self.contrast.get()
        self.brightness = ctk.IntVar(value=1)
        self.old_brightness=self.brightness.get()

        self.min_distance = ctk.IntVar(value=1)
        self.max_distance = ctk.IntVar(value=50)

        self.max_line_gap = ctk.IntVar(value=1)
        self.min_line_length = ctk.IntVar(value=10)
        self.threshold = ctk.IntVar(value=50)

        self.kernel_size_open = ctk.IntVar(value=3)
        self.kernel_size_close = ctk.IntVar(value=3)


        self.kernel_size_erosion = ctk.IntVar(value=3)
        self.iterations = ctk.IntVar(value=1)

        self.dilation_iterations = ctk.IntVar(value=1)
        self.dilation_kernel_size=ctk.IntVar(value=1)

        self.threshold_segmentation = ctk.IntVar(value=0)

        self.filters={"lpf":ctk.IntVar(value=1),
                      "hpf":ctk.IntVar(value=1),
                      "mean":ctk.IntVar(value=1),
                      "median":ctk.IntVar(value=1)}
        #trace change
        self.contrast.trace('w',self.manipulate_contrast_and_brightness)
        self.brightness.trace('w',self.manipulate_contrast_and_brightness)

        self.filters["lpf"].trace('w',self.lpf)
        self.filters["hpf"].trace('w',self.hpf)
        self.filters["mean"].trace('w',self.mean)
        self.filters["median"].trace('w', self.median)

        self.threshold_segmentation.trace('w',self.apply_thresholding_segmentation)
    


        #widgets
        self.image_import_widgets = ImageImport(master=self,import_function=self.import_image)

        self.counter =0


        #mainloop:
        self.mainloop()

    def import_image(self,image_path):
        self.image_import_widgets.grid_forget()

        self.original_image = Image.open(image_path)

        self.original_image_cv = cv2.imread(image_path)

        self.adjusted_image_cv = cv2.imread(image_path)

        self.reset_image = self.original_image_cv

        # self.adjusted_image_cv = cv2.GaussianBlur(self.original_image_cv, (9, 9), 2)
        self.grey_image_cv = cv2.imread(image_path,0)

        self.original_image_cv = cv2.cvtColor(self.original_image_cv,cv2.COLOR_BGR2RGB)
        self.adjusted_image_cv = cv2.cvtColor(self.adjusted_image_cv,cv2.COLOR_BGR2RGB)
        self.adjusted_image_cv_copy = self.adjusted_image_cv

        self.manip_tab = manip_tab(master=self,
                                   contrast_var=self.contrast,
                                   brightness_var=self.brightness,
                                   sobel=self.sobel_edge_detection,
                                   prewitt=self.prewitt_edge_detection,
                                   robert=self.robert_edge_detection,
                                   reset=self.reset, filters=self.filters
                                   ,hough_line=self.hough_line, hough_circle=self.hough_circle,
                                   max_distance=self.max_distance, min_distance=self.min_distance,
                                   iterations=self.iterations,kernel_size_erosion=self.kernel_size_erosion,
                                   erode=self.erosion,dilate=self.dilation,dilation_iterations=self.dilation_iterations,
                                   dilation_kernel_size=self.dilation_kernel_size,reset_filter=self.reset_filters,apply_filters=self.apply_filters,
                                   max_line_gap=self.max_line_gap,min_line_distance=self.min_line_length,threshold=self.threshold,apply_contrast=self.apply_contrast,
                                   kernel_size_close=self.kernel_size_close,kernel_size_open=self.kernel_size_open,
                                   apply_open=self.apply_open,apply_close=self.apply_close,threshold_segmentaion=self.threshold_segmentation,apply_segmentation=self.apply_segmentaion)


        self.image_layout = image_layout(master=self, place_image=self.place_image)

        self.reset_button = ctk.CTkButton(master=self, text='reset', command=self.reset).place(x=850,y=10)

        self.back_btn = ctk.CTkButton(master=self,text="X",command=self.back,width=50).place(x=750,y=10)

        self.save_btn = ctk.CTkButton(master=self,text="SAVE",command=self.save).place(x=600,y=10)

        self.image_tk = ImageTk.PhotoImage(self.original_image)
        self.backup=self.original_image_cv

    def back(self):
        self.manip_tab.grid_forget()
        self.image_layout.grid_forget()
        self.image_import_widgets = ImageImport(master=self,import_function=self.import_image)
    def place_image(self,event):
        self.image_layout.delete("all")
        self.canvas_width = event.width
        self.canvas_height = event.height
        self.image_layout.create_image(self.canvas_width/2,self.canvas_height/2,image=self.image_tk)
        self.resize_image()
    def resize_image(self):
        self.image_aspect = self.original_image.size[0]/self.original_image.size[1]
        self.canvas_aspect = self.canvas_width/self.canvas_height

        if self.canvas_aspect > self.image_aspect:
            self.image_height = int(self.canvas_height)
            self.image_width = int(self.image_aspect*self.image_height)
        else:
            self.image_width = int(self.canvas_width)
            self.image_height = int(self.image_width/self.image_aspect)

        resized_image = self.original_image.resize((self.image_width,self.image_height))
        self.image_tk = ImageTk.PhotoImage(resized_image)
        self.image_layout.create_image(self.canvas_width/2,self.canvas_height/2,image=self.image_tk)

    def manipulate_contrast_and_brightness(self,*args):

        # actual_contrast=self.contrast.get()/self.old_contrast
        # actual_brightness=self.brightness.get()-self.old_brightness

        self.adjusted_image_cv = cv2.convertScaleAbs(self.adjusted_image_cv_copy,alpha=self.contrast.get(),beta=self.brightness.get())


        # convert cv2 image to pil image
        self.original_image = Image.fromarray(self.adjusted_image_cv)


        #place and resize image
        self.image_tk = ImageTk.PhotoImage(self.original_image)




        # self.old_contrast = self.contrast.get()
        # self.old_brightness = self.brightness.get()
        self.resize_image()

    def sobel_edge_detection(self):
        if len(self.adjusted_image_cv_copy.shape) == 3:
            grey_image_cv = cv2.cvtColor(self.adjusted_image_cv_copy, cv2.COLOR_BGR2GRAY)
        else:
            grey_image_cv = self.adjusted_image_cv_copy

        sobel_x = cv2.Sobel(grey_image_cv, cv2.CV_64F, 1, 0, ksize=3)
        sobel_y = cv2.Sobel(grey_image_cv, cv2.CV_64F, 0, 1, ksize=3)


        sobel_x = cv2.convertScaleAbs(sobel_x)
        sobel_y = cv2.convertScaleAbs(sobel_y)

        self.adjusted_image_cv_copy = cv2.addWeighted(sobel_x, 0.5, sobel_y, 0.5, 0)



        #update the original image to be resized
        self.original_image = Image.fromarray(self.adjusted_image_cv_copy)


        # convert the original image to pil format again then place and resize
        self.image_tk = ImageTk.PhotoImage(self.original_image)
        
        self.resize_image()
    def prewitt_edge_detection(self):
        kernel_x = np.array([[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]])
        kernel_y = np.array([[-1, -1, -1], [0, 0, 0], [1, 1, 1]])
        prewitt_x = cv2.filter2D(self.adjusted_image_cv_copy, -1, kernel_x)
        prewitt_y = cv2.filter2D(self.adjusted_image_cv_copy, -1, kernel_y)
        prewitt_x = cv2.convertScaleAbs(prewitt_x)
        prewitt_y = cv2.convertScaleAbs(prewitt_y)

        self.adjusted_image_cv_copy = cv2.addWeighted(prewitt_x, 0.5, prewitt_y, 0.5, 0)


        self.original_image = Image.fromarray(self.adjusted_image_cv_copy)
        self.image_tk = ImageTk.PhotoImage(self.original_image)
    
        self.resize_image()


        
    def robert_edge_detection(self):
        kernel_x = np.array([[1,0],[0, -1]])
        kernel_y = np.array([[0,1],[-1, 0]])
        robert_x = cv2.filter2D(self.adjusted_image_cv_copy, -1, kernel_x)
        robert_y = cv2.filter2D(self.adjusted_image_cv_copy, -1, kernel_y)
        robert_x = cv2.convertScaleAbs(robert_x)
        robert_y = cv2.convertScaleAbs(robert_y)
        self.adjusted_image_cv_copy = cv2.addWeighted(robert_x, 0.5, robert_y, 0.5, 0)


        self.original_image = Image.fromarray(self.adjusted_image_cv_copy)
        self.image_tk = ImageTk.PhotoImage(self.original_image)
        self.resize_image()


    def lpf(self,*args):

        kernel_size = (9, 9)
        sigma_x = self.filters["lpf"].get()

        self.adjusted_image_cv = cv2.GaussianBlur(self.adjusted_image_cv_copy, kernel_size, sigma_x)

        # update the original image to be resized
        self.original_image = Image.fromarray(self.adjusted_image_cv)

        # convert the original image to pil format again then place and resize
        self.image_tk = ImageTk.PhotoImage(self.original_image)
        self.resize_image()
    def hpf(self,*args):

        hpf_kernel_size = (9, 9)
        strength =self.filters["hpf"].get()

        blurred = cv2.GaussianBlur(self.adjusted_image_cv_copy, hpf_kernel_size, sigmaX=0)

        # Subtract the blurred image from the original image to get the high-pass filter effect
        high_pass = cv2.subtract(self.adjusted_image_cv_copy, blurred)

        # Increase the strength of the high-pass filter effect
        high_pass = cv2.multiply(high_pass, strength)

        # Add the high-pass filtered image to the original image to enhance the high-frequency components
        enhanced = cv2.add(self.adjusted_image_cv_copy, high_pass)

        self.adjusted_image_cv = enhanced

        self.original_image = Image.fromarray(self.adjusted_image_cv)
        self.image_tk = ImageTk.PhotoImage(self.original_image)
        self.resize_image()

    def median(self,*args):

        median_kernal_size = self.filters["median"].get()
        median_filter=cv2.medianBlur(self.adjusted_image_cv_copy, median_kernal_size)
        self.adjusted_image_cv = median_filter

        self.original_image = Image.fromarray(self.adjusted_image_cv)
        self.image_tk = ImageTk.PhotoImage(self.original_image)
        self.resize_image()
    def mean(self,*args):

        mean_kernal_size=self.filters["mean"].get()
        mean_filter=cv2.blur(self.adjusted_image_cv_copy,(mean_kernal_size,mean_kernal_size))
        self.adjusted_image_cv = mean_filter
        self.original_image = Image.fromarray(self.adjusted_image_cv)
        self.image_tk = ImageTk.PhotoImage(self.original_image)
        self.resize_image()
        
    def hough_line(self):
        if len(self.adjusted_image_cv.shape) == 3:
            grey_image_cv = cv2.cvtColor(self.adjusted_image_cv, cv2.COLOR_BGR2GRAY)
        else:
            grey_image_cv = self.adjusted_image_cv

        # Apply Canny edge detection
        edges = cv2.Canny(grey_image_cv, 50, 150, apertureSize=3)

        # Perform Hough line detection
        lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=self.threshold.get(), minLineLength=self.min_line_length.get(),
                                maxLineGap=self.max_line_gap.get())
        color = [0, 0, 0]

        if len(self.adjusted_image_cv_copy.shape) == 3 and self.adjusted_image_cv_copy.shape[2] == 3:
            r, g, b = cv2.split(self.adjusted_image_cv_copy)

            # Count the number of pixels in each color channel
            red_pixels = np.sum(r)
            green_pixels = np.sum(g)
            blue_pixels = np.sum(b)

            if red_pixels > green_pixels and red_pixels > blue_pixels:
                color[2] = 255
            elif green_pixels > red_pixels and green_pixels > blue_pixels:
                color[0] = 255
            else:
                color[0] = 255
        else:
            color[0] = 255

        color = tuple(color)


        # Draw detected lines on the image
        for line in lines:
            x1, y1, x2, y2 = line[0]
            cv2.line(self.adjusted_image_cv_copy, (x1, y1), (x2, y2), color, 2)

        self.original_image = Image.fromarray(self.adjusted_image_cv_copy)
        self.image_tk = ImageTk.PhotoImage(self.original_image)

        # reset the hough line slider
        self.threshold.set(50)
        self.max_line_gap.set(1)
        self.min_line_length.set(10)

        self.resize_image()

    def hough_circle(self):
        if len(self.adjusted_image_cv.shape) == 3:
            grey_image_cv = cv2.cvtColor(self.adjusted_image_cv, cv2.COLOR_BGR2GRAY)
        else:
            grey_image_cv = self.adjusted_image_cv
        blurred_image = cv2.GaussianBlur(grey_image_cv, (5, 5), 0)

        # Apply Hough Circle Transform
        circles = cv2.HoughCircles(
            blurred_image,
            cv2.HOUGH_GRADIENT,
            dp=1,
            minDist=50,
            param1=200,
            param2=30,
            minRadius=self.min_distance.get(),
            maxRadius=self.max_distance.get()
            )

        color = [0, 0, 0]

        if len(self.adjusted_image_cv_copy.shape) == 3 and self.adjusted_image_cv_copy.shape[2] == 3:
            r, g, b = cv2.split(self.adjusted_image_cv_copy)

            # Count the number of pixels in each color channel
            red_pixels = np.sum(r)
            green_pixels = np.sum(g)
            blue_pixels = np.sum(b)

            if red_pixels > green_pixels and red_pixels > blue_pixels:
                color[2] = 255
            elif green_pixels > red_pixels and green_pixels > blue_pixels:
                color[0] = 255
            else:
                color[0] = 255
        else:
            color[0] = 255

        color = tuple(color)

        # Draw the detected circles on the image
        if circles is not None:
            circles = np.round(circles[0, :]).astype(int)
            for (x, y, r) in circles:
                cv2.circle(self.adjusted_image_cv_copy, (x, y), r, color, 5)

        #update the original image to be resized
        self.original_image = Image.fromarray(self.adjusted_image_cv_copy)
        self.image_tk = ImageTk.PhotoImage(self.original_image)


        # reset the hough circle sliders
        self.min_distance.set(1)
        self.max_distance.set(50)

        self.resize_image()

    def erosion(self):

        if len(self.adjusted_image_cv.shape) == 3:
            grey_image_cv = cv2.cvtColor(self.adjusted_image_cv, cv2.COLOR_BGR2GRAY)
        else:
            grey_image_cv = self.adjusted_image_cv

        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (self.kernel_size_erosion.get(), self.kernel_size_erosion.get()))

        # Apply erosion
        self.adjusted_image_cv = cv2.erode(grey_image_cv, kernel, iterations=self.iterations.get())

        self.adjusted_image_cv_copy = self.adjusted_image_cv

        self.kernel_size_erosion.set(3)
        self.iterations.set(1)

        self.original_image = Image.fromarray(self.adjusted_image_cv_copy)
        self.image_tk = ImageTk.PhotoImage(self.original_image)
        self.resize_image()
    def dilation(self):
        if len(self.adjusted_image_cv.shape) == 3:
            gray_image = cv2.cvtColor(self.adjusted_image_cv, cv2.COLOR_BGR2GRAY)
        else:
            gray_image = self.adjusted_image_cv


        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (self.dilation_kernel_size.get(), self.dilation_kernel_size.get()))

        # Apply dilation
        self.adjusted_image_cv = cv2.dilate(gray_image, kernel, iterations=self.dilation_iterations.get())

        self.adjusted_image_cv_copy = self.adjusted_image_cv

        self.dilation_kernel_size.set(3)
        self.dilation_iterations.set(1)

        self.update_image(self.adjusted_image_cv_copy)

    def apply_open(self):
        kernel_size = self.kernel_size_open.get()
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_size, kernel_size))
        self.adjusted_image_cv = cv2.morphologyEx(self.adjusted_image_cv_copy, cv2.MORPH_OPEN, kernel)

        self.adjusted_image_cv_copy = self.adjusted_image_cv

        self.original_image = Image.fromarray(self.adjusted_image_cv_copy)
        self.image_tk = ImageTk.PhotoImage(self.original_image)

        self.kernel_size_open.set(3)
        self.kernel_size_close.set(3)

        self.resize_image()

    def apply_close(self):
        kernel_size = self.kernel_size_close.get()

        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_size, kernel_size))
        self.adjusted_image_cv = cv2.morphologyEx(self.adjusted_image_cv_copy, cv2.MORPH_CLOSE, kernel)

        self.adjusted_image_cv_copy = self.adjusted_image_cv

        self.original_image = Image.fromarray(self.adjusted_image_cv_copy)
        self.image_tk = ImageTk.PhotoImage(self.original_image)

        self.kernel_size_open.set(3)
        self.kernel_size_close.set(3)

        self.resize_image()

    def apply_thresholding_segmentation(self, *args):
        threshold_value = self.threshold_segmentation.get()
        if len(self.adjusted_image_cv_copy.shape) == 3:
            gray_image = cv2.cvtColor(self.adjusted_image_cv_copy, cv2.COLOR_BGR2GRAY)
        else:
            gray_image = self.adjusted_image_cv

        _,self.adjusted_image_cv = cv2.threshold(gray_image, threshold_value, 255, cv2.THRESH_BINARY)



        self.original_image = Image.fromarray(self.adjusted_image_cv)
        self.image_tk = ImageTk.PhotoImage(self.original_image)
        self.resize_image()



    def reset_filters(self,filter_name,event):
        for filter in self.filters:
            if filter is not filter_name:
                self.filters[filter].set(1)

    def reset(self):
        self.adjusted_image_cv_copy = self.original_image_cv.copy()
        print("reset is clicked")
        self.contrast.set(1)
        self.brightness.set(1)
        for filter in self.filters:
            self.filters[filter].set(1)
        self.original_image = Image.fromarray(self.adjusted_image_cv_copy)
        self.image_tk = ImageTk.PhotoImage(self.original_image)
        self.resize_image()

    def update_image(self,image):
        self.original_image = Image.fromarray(image)
        self.image_tk = ImageTk.PhotoImage(self.original_image)
        self.resize_image()

    def apply_filters(self):
        self.adjusted_image_cv_copy = self.adjusted_image_cv
        for filter in self.filters:
            self.filters[filter].set(1)

    def apply_segmentaion(self):
        self.adjusted_image_cv_copy = self.adjusted_image_cv
        self.threshold_segmentation.set(1)

    def apply_contrast(self):
        self.adjusted_image_cv_copy=self.adjusted_image_cv
        self.contrast.set(1)
        self.brightness.set(1)
    def save(self):
        save_path = ctk.filedialog.asksaveasfile(defaultextension=".jpg")

        print(save_path)


        if len(self.adjusted_image_cv_copy.shape) == 2 or self.adjusted_image_cv_copy.shape[2] == 1:
            self.adjusted_image_cv_copy = cv2.cvtColor(self.adjusted_image_cv_copy, cv2.COLOR_GRAY2RGB)

        self.adjusted_image_cv_copy = cv2.cvtColor(self.adjusted_image_cv_copy,cv2.COLOR_BGR2RGB)

        if save_path:
            cv2.imwrite(str(save_path.name),self.adjusted_image_cv_copy)




App()