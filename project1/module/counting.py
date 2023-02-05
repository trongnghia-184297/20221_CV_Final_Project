import os
import numpy as np
import cv2

class Counting:
    def __init__(self, path_in_origin , path_in_denoise, path_out, step):
        self.step = step
        self.path_in_origin = path_in_origin
        self.path_in_denoise = path_in_denoise
        self.path_out = path_out

    def create_folder(self, path_out):
        if not os.path.exists(path_out):
            os.mkdir(path_out)

    def average_list(self, lst):
        return sum(lst)/len(lst)

    def get_shape(self, img):
        height = img.shape[0]
        width = img.shape[1]
        channels = img.shape[2]

        return height, width

    def apply_adaptive_threshold(self, img):
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        thresh_adaptive = 180
        kernelSize = 25
        C = -10
        img_adapt_thresh = cv2.adaptiveThreshold(gray,thresh_adaptive,cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY,kernelSize, C)
        return img_adapt_thresh

    def apply_opening(self, img):
        kernelSize = (4, 4)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, kernelSize)
        # print(kernel)
        opening_img = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel)
        return opening_img

    def get_contour(self, img):
        contours, _ = cv2.findContours(img,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
        return contours

    def visualize_contours(self, img, contours):
        for item in contours:
            # Normal Rectangle
            x,y,w,h = cv2.boundingRect(item)
            
            # Oriented rectangle
            rect = cv2.minAreaRect(item)
            box = cv2.boxPoints(rect)
            box = np.int0(box)
            
            # Normal rectangle
            # img = cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,0),2)

            # Bounding contour
            img = cv2.drawContours(img,[item],0,(255,0,0),2)

            # Small rectangle
            img = cv2.drawContours(img,[box],0,(0,255,255),2)

        return img

    def filter_connected_objects(self, img, contours, thres_connected):
        for item in contours:
            # Normal Rectangle
            x,y,w,h = cv2.boundingRect(item)
            
            # Area of contour
            area_contour = cv2.contourArea(item)
            
            # Oriented rectangle
            rect = cv2.minAreaRect(item)
            box = cv2.boxPoints(rect)
            box = np.int0(box)
            area_rec = cv2.contourArea(box)
            
            if area_contour/area_rec > thres_connected:
                continue
            else:
                img_copy = img[y:y+h,x:x+w]
                kernelSize = (7, 9)
                kernel = cv2.getStructuringElement(cv2.MORPH_RECT, kernelSize)
                img_copy = cv2.morphologyEx(img_copy, cv2.MORPH_OPEN, kernel)
                img[y:y+h,x:x+w] = img_copy

        return img        

    def filter_small_objects(self, new_contours, thres_area):

        area_list = []
        extracted_contours = []

        for contour in new_contours:
            area_contour = cv2.contourArea(contour)
            area_list.append(area_contour)

        mean_area = self.average_list(area_list)

        for item in new_contours:
            # Area of contour
            area_contour = cv2.contourArea(item)

            # Oriented rectangle
            rect = cv2.minAreaRect(item)
            box = cv2.boxPoints(rect)
            box = np.int0(box)
            
            if area_contour/mean_area > thres_area:
                extracted_contours.append(item)

        return extracted_contours, len(extracted_contours)


    def window_split(self, list_window, step):
        final_split_windows = []
        windows_sametime = []
        for idx, item in enumerate(list_window):
            if len(windows_sametime) == step:
                final_split_windows.append(windows_sametime)
                mutual_window = windows_sametime[-1]
                windows_sametime = []
                windows_sametime.append(mutual_window)
            windows_sametime.append(item)

            if idx == len(list_window) - 1:
                final_split_windows.append(windows_sametime)

        return final_split_windows
    # Example:
    # list = [1,2,3,4,5,6,7], step = 2 -> return [[1,2],[2,3],[3,4],[4,5],[5,6],[6,7]]
    # list = [1,2,3,4,5,6,7], step = 3 -> return [[1,2,3],[3,4,5],[5,6,7]]
    # list = [1,2,3,4,5,6,7], step = 4 -> return [[1,2,3,4],[4,5,6,7]]


    def window_visualization(self, list_window, list_name, step):
        if step == 1:
            for idx, item in enumerate(list_window):
                size0 = (640,640)
                pos0 = (600,200)
                item = cv2.resize(item, size0)
                cv2.imshow(list_name[idx],item)
                cv2.moveWindow(list_name[idx], pos0[0], pos0[1])
                
                cv2.waitKey(0)
                cv2.destroyAllWindows()
                
        else:
            # Config step = 2
            size2 = (600,600)
            pos20 = [300,250]
            pos21 = [1000,250]

            # Config step = 3
            size3 = (540,540)
            pos30 = (70,250)
            pos31 = (690,250)
            pos32 = (1310,250)

            # Config step = 4
            size4 = (440,440)

            # 1 approach
            pos40 = (430,20)
            pos41 = (1000,20)
            pos42 = (430,560)
            pos43 = (1000,560)

            # 2 approach
            pos40 = (30,280)
            pos41 = (500,280)
            pos42 = (970,280)
            pos43 = (1440,280)

            list_window_separate = self.window_split(list_window, step)
            list_name_separate = self.window_split(list_name, step)
            for idx1, windows in enumerate(list_window_separate):
                for idx2, window in enumerate(windows):
                    name_window = list_name_separate[idx1][idx2]
                    if step == 2:
                        window = cv2.resize(window, size2) 
                        cv2.imshow(name_window, window)
                        if idx2 == 0:
                            cv2.moveWindow(name_window, pos20[0], pos20[1])
                        elif idx2 == 1:
                            cv2.moveWindow(name_window, pos21[0], pos21[1])

                    elif step == 3:
                        window = cv2.resize(window, size3) 
                        cv2.imshow(name_window, window)
                        if idx2 == 0:
                            cv2.moveWindow(name_window, pos30[0], pos30[1])
                        elif idx2 == 1:
                            cv2.moveWindow(name_window, pos31[0], pos32[1])
                        elif idx2 == 2:
                            cv2.moveWindow(name_window, pos32[0], pos32[1])
                    elif step == 4:
                        window = cv2.resize(window, size4) 
                        cv2.imshow(name_window, window)
                        if idx2 == 0:
                            cv2.moveWindow(name_window, pos40[0], pos40[1])
                        elif idx2 == 1:
                            cv2.moveWindow(name_window, pos41[0], pos41[1])
                        elif idx2 == 2:
                            cv2.moveWindow(name_window, pos42[0], pos42[1])
                        elif idx2 == 3:
                            cv2.moveWindow(name_window, pos43[0], pos43[1])

                cv2.waitKey(0)
                cv2.destroyAllWindows()


    def saving(self, img):
        # Saving
        name_list = self.path_in_denoise.split("/")
        name_png = name_list[-1]
        path_out_png = self.path_out + "/" + name_png
        cv2.imwrite(path_out_png, img)


    def counting_object(self):
        list_window_name = [
            "Step 1: Original image",
            "Step 2: Denoise image",
            "Step 3: Applying adaptive thresholding",
            "Step 4: Applying opening operation",
            "Step 5: Unfiltered contours",
            "Step 6: Filtered contours",
            "Step 7: Final Counting"
        ]

        list_window = []

        # Create destination folder
        self.create_folder(self.path_out)

        # Read original image
        img_origin = cv2.imread(self.path_in_origin)
        list_window.append(img_origin)
        # cv2.imshow("Original image",img_origin)

        # Read denoise image
        img = cv2.imread(self.path_in_denoise)
        list_window.append(img)
        # cv2.imshow("Denoise image",img)

        # Apply adaptive thresholding
        adaptive_img = self.apply_adaptive_threshold(img)
        list_window.append(adaptive_img)
        # cv2.imshow("Applying adaptive thresholding",adaptive_img)

        # Apply opening operation
        opening_img = self.apply_opening(adaptive_img)
        list_window.append(opening_img)
        # cv2.imshow("Applying opening operation",opening_img)

        # Getting unfiltered contours
        unfilter_contours = self.get_contour(opening_img)
        unfilter_contours_img = img.copy()
        unfilter_contours_img =self.visualize_contours(unfilter_contours_img, unfilter_contours)
        list_window.append(unfilter_contours_img)
        # cv2.imshow("Unfilter contours",unfilter_contours_img)

        # Filtering contours
        # Filter connected objects
        connected_filter_img = self.filter_connected_objects(opening_img, unfilter_contours, 0.5)
        connected_filter_contours = self.get_contour(connected_filter_img)
        # Filter small objects
        final_filter_contours, num_objects = self.filter_small_objects(connected_filter_contours, 0.18)
        filter_contours_img = img.copy()
        filter_contours_img = self.visualize_contours(filter_contours_img, final_filter_contours)
        list_window.append(filter_contours_img)
        # cv2.imshow("Filter contours",filter_contours_img)

        # FINAL RESULTS
        final_contours_img = img_origin.copy()
        final_contours_img = self.visualize_contours(final_contours_img, final_filter_contours)

        # Visualize text
        h_img, w_img = self.get_shape(img)
        font = cv2.FONT_HERSHEY_SIMPLEX
        position_1 = (w_img-40, h_img-25)
        position_2 = (w_img-35, h_img-10)
        fontScale  = 0.5
        fontColor  = (0,0,255)
        thickness  = 2

        cv2.putText(final_contours_img,'Total', position_1, font, fontScale, fontColor, thickness)
        cv2.putText(final_contours_img,str(num_objects), position_2, font, fontScale, fontColor, thickness)
        list_window.append(final_contours_img)

        # Showing final result
        size_final = (640,640)
        pos_final = (600,200)
        final_contours_img_resize = cv2.resize(final_contours_img, size_final)
        cv2.imshow("Final Counting",final_contours_img_resize)
        cv2.moveWindow("Final Counting", pos_final[0], pos_final[1])
        
        cv2.waitKey(0)
        cv2.destroyAllWindows()

        # Print number of object
        print("Object counting using finding contour = " + str(num_objects)) 

        # Process visualization
        self.window_visualization(list_window, list_window_name, int(self.step))

        # Saving img of final result
        self.saving(final_contours_img)


# counting_test = Counting('denoise/noise_periodic.png', 'out', 1)
# counting_test = Counting('denoise/noise_pepper.png', 'out', 1)
# counting_test = Counting('denoise/noise_connected.png', 'out', 0)
# counting_test = Counting('denoise/normal.png', 'out', 1)
# counting_test.counting_object()