###################Tkinter imports#####################################
from tkinter import *
from tkinter import ttk
import cv2
from PIL import Image, ImageTk
import datetime
from datetime import date
from tkinter import filedialog
import os
import numpy as np
from tkinter.scrolledtext import ScrolledText
import tkinter.font as font
import time
global counter_2
global counter
counter=0
global opq
global o
global op
global coor
global counter_111
global defect
global all_defects
#################Yolo V5 imports#########################################
import argparse
import os
import sys
from pathlib import Path
import numpy as np
from matplotlib import pyplot as plt
import shutil
import cv2
import torch
import torch.backends.cudnn as cudnn
import argparse
import os

import sys
from pathlib import Path
import glob
global counter_11
import cv2
import torch
import torch.backends.cudnn as cudnn
global counter_1
FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # YOLOv5 root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative
from models.common import DetectMultiBackend
from utils.datasets import IMG_FORMATS, VID_FORMATS, LoadImages, LoadStreams
from utils.general import (LOGGER, check_file, check_img_size, check_imshow, check_requirements, colorstr,
                           increment_path, non_max_suppression, print_args, scale_coords, strip_optimizer, xyxy2xywh)
from utils.plots import Annotator, colors, save_one_box
from utils.torch_utils import select_device, time_sync
import cv2
import numpy as np

global number_defects

import glob
from PIL import Image
from pydicom import dcmread
from pydicom.data import get_testdata_file
import os
from PIL import Image, ImageFilter, ImageEnhance,ImageOps
import cv2
import matplotlib.pyplot as plt
from scipy.ndimage.filters import median_filter
import numpy as np




def unsharp(image, sigma, strength):

    # Median filtering
    image_mf = cv2.medianBlur(image, sigma)

    # Calculate the Laplacian
    lap = cv2.Laplacian(image_mf)

    # Calculate the sharpened image
    sharp = image-strength*lap

    # Saturate the pixels in either direction
    sharp[sharp>255] = 255
    sharp[sharp<0] = 0
    
    return sharp







# Search files with .txt extension in current directory
'''pattern = "D:\\latest pasted\\GRSE_TEST\\*"
files = glob.glob(pattern)

# deleting the files with txt extension
for file in files:
    os.remove(file)
path = sorted( filter( os.path.isfile,
                        glob.glob("D:\\new download\GRSE\\3023 - RT\\3023 - Erection\\3023 - A2 + A3\\3023 - A2KFPS + A3KFC\\*.dcm") ) )
#filename = get_testdata_file("D:\\latest pasted\\drive-download-20220203T183947Z-001\\3020 - Pipe - 1F1 - 010 -  40 - X3 - A - 24.12.2019.dcm")'''
def stretch1(a, lower_thresh, upper_thresh):
    r = 65535.0/(upper_thresh-lower_thresh+2) # unit of stretching
    out = np.round(r*np.where(a>=lower_thresh,a-lower_thresh+1,0)).clip(max=65535)
    return out.astype(a.dtype)

def stretch2(a, lower_thresh, upper_thresh):
    r = 255.0/(upper_thresh-lower_thresh+2) # unit of stretching
    out = np.round(r*np.where(a>=lower_thresh,a-lower_thresh+1,0)).clip(max=255)
    return out.astype(a.dtype)
from openpyxl import Workbook
from openpyxl import load_workbook
from openpyxl.styles import Alignment  

def excel():
    global all_defects
    global namess
    workbook = load_workbook(filename="D:\\new download\\TEST.xlsx")  
    work_sheet = workbook.active  
      
    # cells to merge
      
    for i in range(0, (len(namess)-1)):
        work_sheet['B'+str(17+3*i)]=namess[i]
        
    for i in range(0, (len(all_defects)-1)):
        if (all_defects[i][0]== ''):
            work_sheet['G'+str(17+3*i)]="Non-defective film"
        else:
            work_sheet['G'+str(17+3*i)]="Defective film" + "\n" + all_defects[i][0]
        
        
    work_sheet.delete_rows(17+3*len(namess),180-3*len(namess))
      
    # save the workbook  3*len(namess)
    workbook.save(filename="D:\\new download\\TEST_1.xlsx")
    lbl14.configure(text="Saved as: D:\\new download\\TEST_1.xlsx", font="bold")
#ds = dcmread("D:\\latest pasted\\drive-download-20220203T183947Z-001\\3022 - B1KAC - J 18 - BS-P-03.11.19.dcm")
import numpy as np
import joblib
#filename = "C:\\Users\\user\\Desktop\\finalized_model.sav"
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
 
# some time later...
 
# load the model from disk
#loaded_model = joblib.load(filename)
from skimage import exposure
import numpy as np
def histogram_equalize(img):
    img = rgb2gray(img)
    img_cdf, bin_centers = exposure.cumulative_distribution(img)
    return np.interp(img, bin_centers, img_cdf)




@torch.no_grad()
def run(weights=ROOT / 'yolov5s.pt',  # model.pt path(s)
        source=ROOT / 'data/images',  # file/dir/URL/glob, 0 for webcam
        data=ROOT / 'data/coco128.yaml',  # dataset.yaml path
        imgsz=(416, 416),  # inference size (height, width)
        conf_thres=0.25,  # confidence threshold
        iou_thres=0.45,  # NMS IOU threshold
        max_det=1000,  # maximum detections per image
        device='',  # cuda device, i.e. 0 or 0,1,2,3 or cpu
        view_img=False,  # show results
        save_txt=False,  # save results to *.txt
        save_conf=False,  # save confidences in --save-txt labels
        save_crop=False,  # save cropped prediction boxes
        nosave=False,  # do not save images/videos
        classes=None,  # filter by class: --class 0, or --class 0 2 3
        agnostic_nms=False,  # class-agnostic NMS
        augment=False,  # augmented inference
        visualize=False,  # visualize features
        update=False,  # update all models
        project=ROOT / 'runs/detect',  # save results to project/name
        name='exp',  # save results to project/name
        exist_ok=False,  # existing project/name ok, do not increment
        line_thickness=1,  # bounding box thickness (pixels)
        hide_labels=False,  # hide labels
        hide_conf=False,  # hide confidences
        half=False,  # use FP16 half-precision inference
        dnn=False,  # use OpenCV DNN for ONNX inference
        ):
    source = str(source)
    save_img = not nosave and not source.endswith('.txt')  # save inference images
    is_file = Path(source).suffix[1:] in (IMG_FORMATS + VID_FORMATS)
    is_url = source.lower().startswith(('rtsp://', 'rtmp://', 'http://', 'https://'))
    webcam = source.isnumeric() or source.endswith('.txt') or (is_url and not is_file)
    if is_url and is_file:
        source = check_file(source)  # download
    global all_defects
    global namess
    global counter_111
    all_defects=[]
    namess=[]
    global defect
    global opq
    opq=[]
    defect=[]
    global number_defects
    number_defects=[]
    global coor
    global op
    op=[]
    coor=[]
    # Directories
    save_dir = increment_path(Path(project) / name, exist_ok=exist_ok)  # increment run
    (save_dir / 'labels' if save_txt else save_dir).mkdir(parents=True, exist_ok=True)  # make dir
    global counter_1
    counter_1=0
    k=0
    # Load model
    device = select_device(device)
    model = DetectMultiBackend(weights, device=device, dnn=dnn, data=data)
    stride, names, pt, jit, onnx, engine = model.stride, model.names, model.pt, model.jit, model.onnx, model.engine
    imgsz = check_img_size(imgsz, s=stride)  # check image size
    global counter
    counter=0
    # Half
    half &= (pt or jit or onnx or engine) and device.type != 'cpu'  # FP16 supported on limited backends with CUDA
    if pt or jit:
        model.model.half() if half else model.model.float()

    # Dataloader
    if webcam:
        view_img = check_imshow()
        cudnn.benchmark = True  # set True to speed up constant image size inference
        dataset = LoadStreams(source, img_size=imgsz, stride=stride, auto=pt)
        bs = len(dataset)  # batch_size
    else:
        dataset = LoadImages(source, img_size=imgsz, stride=stride, auto=pt)
        bs = 1  # batch_size
    vid_path, vid_writer = [None] * bs, [None] * bs
    
    global fln
    
    #all_defects = [[]] * len(fln)
    # Run inference
    model.warmup(imgsz=(1, 3, *imgsz), half=half)  # warmup
    dt, seen = [0.0, 0.0, 0.0], 0
    for path, im, im0s, vid_cap, s in dataset:
        t1 = time_sync()
        im = torch.from_numpy(im).to(device)
        im = im.half() if half else im.float()  # uint8 to fp16/32
        im /= 255  # 0 - 255 to 0.0 - 1.0
        if len(im.shape) == 3:
            im = im[None]  # expand for batch dim
        t2 = time_sync()
        dt[0] += t2 - t1
        counter_111=counter_111+1
        a=(100/len(fln))*counter_111
        progress_detect['value']=a
        f1.update_idletasks()
        time.sleep(0.0000001)
        b= "% s" % a
        #time.sleep(.0000005)
        lbl13.configure(text="Detecting :" + b +"%", font="bold")
        coor.append(["nothing"])

        # Inference
        visualize = increment_path(save_dir / Path(path).stem, mkdir=True) if visualize else False
        pred = model(im, augment=augment, visualize=visualize)
        t3 = time_sync()
        dt[1] += t3 - t2

        # NMS
        pred = non_max_suppression(pred, conf_thres, iou_thres, classes, agnostic_nms, max_det=max_det)
        dt[2] += time_sync() - t3

        # Second-stage classifier (optional)
        # pred = utils.general.apply_classifier(pred, classifier_model, im, im0s)

        # Process predictions
        
        for i, det in enumerate(pred):  # per image
            seen += 1
            if webcam:  # batch_size >= 1
                p, im0, frame = path[i], im0s[i].copy(), dataset.count
                s += f'{i}: '
            else:
                p, im0, frame = path, im0s.copy(), getattr(dataset, 'frame', 0)

            p = Path(p)  # to Path
            save_path = str(save_dir / p.name)  # im.jpg
            txt_path = str(save_dir / 'labels' / p.stem) + ('' if dataset.mode == 'image' else f'_{frame}')  # im.txt
            s += '%gx%g ' % im.shape[2:]  # print string
            gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
            imc = im0.copy() if save_crop else im0  # for save_crop
            annotator = Annotator(im0, line_width=line_thickness, example=str(names))
            
            if (len(det)>0):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_coords(im.shape[2:], det[:, :4], im0.shape).round()
                coor.pop()
                coor.append(det[:, :4])
                if((det[:, :4].shape[1])==4):
                    counter_1=counter_1+1


            
                    
                        
                

                
                # Print results
                for c in det[:, -1].unique():
                    n = (det[:, -1] == c).sum()  # detections per class
                    s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "  # add to string
                    #all_defects.append([f"{n} {names[int(c)]}, "])

                     
                
                
                
                

            
                # Write results
                for *xyxy, conf, cls in reversed(det):
                    if save_txt:  # Write to file
                        xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
                        line = (cls, *xywh, conf) if save_conf else (cls, *xywh)  # label format
                        with open(txt_path + '.txt', 'a') as f:
                            f.write(('%g ' * len(line)).rstrip() % line + '\n')

                    if save_img or save_crop or view_img:  # Add bbox to image
                        c = int(cls)  # integer class
                        label = None if hide_labels else (names[c] if hide_conf else f'{names[c]} {conf:.2f}')
                        annotator.box_label(xyxy, label, color=colors(c, True))
                        if save_crop:
                            save_one_box(xyxy, imc, file=save_dir / 'crops' / names[c] / f'{p.stem}.jpg', BGR=True)
            print(s)
            a=s.split('416')[0]
            b=a.split(':')[1]
            c=b.split(':')[0]
            d=os.path.basename(c)
            e=s.split('416')[1]
            f=e[4:]
            test_string=f[:-2]
            res = [int(i) for i in test_string.split() if i.isdigit()]
            number_defects.append(res)
            #name.append([c])
            all_defects.append([f[:-2]])
            namess.append(d)
            
            # Print time (inference-only)
            LOGGER.info(f'{s}Done. ({t3 - t2:.3f}s)')
            # Stream results
            im0 = annotator.result()
            if view_img:
                cv2.imshow(str(p), im0)
                cv2.waitKey(1)  # 1 millisecond

            # Save results (image with detections)
            if save_img:
                if dataset.mode == 'image':
                    cv2.imwrite(save_path, im0)
                else:  # 'video' or 'stream'
                    if vid_path[i] != save_path:  # new video
                        vid_path[i] = save_path
                        if isinstance(vid_writer[i], cv2.VideoWriter):
                            vid_writer[i].release()  # release previous video writer
                        if vid_cap:  # video
                            fps = vid_cap.get(cv2.CAP_PROP_FPS)
                            w = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                            h = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                        else:  # stream
                            fps, w, h = 30, im0.shape[1], im0.shape[0]
                        save_path = str(Path(save_path).with_suffix('.mp4'))  # force *.mp4 suffix on results videos
                        vid_writer[i] = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
                    vid_writer[i].write(im0)
    # Print results
        
    
    t = tuple(x / seen * 1E3 for x in dt)  # speeds per image
    LOGGER.info(f'Speed: %.1fms pre-process, %.1fms inference, %.1fms NMS per image at shape {(1, 3, *imgsz)}' % t)
    if save_txt or save_img:
        s = f"\n{len(list(save_dir.glob('labels/*.txt')))} labels saved to {save_dir / 'labels'}" if save_txt else ''
        print(s)
        LOGGER.info(f"Results saved to {colorstr('bold', save_dir)}{s}")
    if update:
        strip_optimizer(weights)  # update model (to fix SourceChangeWarning)
    global o
    o=[]
    print(all_defects)
    print(namess)
    print(coor)
    print(number_defects)
    import re
    for m in all_defects:
       z=m[0].split()
       l=[]
       for i in  range(len(z)):
           if z[i].isnumeric()== True:
               for w in range (int(z[i])):
                   l.append(z[i+1])
           i=i+2
       o.append(l)
               
    o = [[j.strip(',') for j in i] for i in o]      
    for i in o:
        opq.append(str(list(set(i))))
    for m in range(len(o)):
        n=o[m]
        for p in range(len(n)):
            q=n[p]
            
            if(q=='blowholes' ):
                W= np.array(coor[m][p])
                M= abs(W[0]-W[2])
                N=abs(W[1]-W[3])
                print(N/M)
                if 1.2<(N/M)<2:
                    o[m][p]="Piping/blowholes"
                if (N/M)>3:
                    o[m][p]="porosotiess"
            if(q=='LOF' ):
                W= np.array(coor[m][p])
                M= abs(W[0]-W[1])
                N=abs(W[1]-W[3])
                if 1.5<(N/M)<2:
                    o[m][p]="Inclusion"
                
                

    print(o)
    for i in o:
        op.append(str(list(set(i))))
        
            

            
                
          
                
                  
                  
            
            

            
    
    

    
    
def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', nargs='+', type=str, default=ROOT / "weld.pt", help='model path(s)')
    parser.add_argument('--source', type=str, default=ROOT / "images\\", help='file/dir/URL/glob, 0 for webcam')
    parser.add_argument('--data', type=str, default=ROOT / 'data/coco128.yaml', help='(optional) dataset.yaml path')
    parser.add_argument('--imgsz', '--img', '--img-size', nargs='+', type=int, default=[416], help='inference size h,w')
    parser.add_argument('--conf-thres', type=float, default=0.001, help='confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.002, help='NMS IoU threshold')
    parser.add_argument('--max-det', type=int, default=1000, help='maximum detections per image')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--view-img', action='store_true', help='show results')
    parser.add_argument('--save-txt', action='store_true', help='save results to *.txt')
    parser.add_argument('--save-conf', action='store_true', help='save confidences in --save-txt labels')
    parser.add_argument('--save-crop', action='store_true', help='save cropped prediction boxes')
    parser.add_argument('--nosave', action='store_true', help='do not save images/videos')
    parser.add_argument('--classes', nargs='+', type=int, help='filter by class: --classes 0, or --classes 0 2 3')
    parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
    parser.add_argument('--augment', action='store_true', help='augmented inference')
    parser.add_argument('--visualize', action='store_true', help='visualize features')
    parser.add_argument('--update', action='store_true', help='update all models')
    parser.add_argument('--project', default=ROOT / 'runs/detect', help='save results to project/name')
    parser.add_argument('--name', default='exp', help='save results to project/name')
    parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
    parser.add_argument('--line-thickness', default=3, type=int, help='bounding box thickness (pixels)')
    parser.add_argument('--hide-labels', default=False, action='store_true', help='hide labels')
    parser.add_argument('--hide-conf', default=False, action='store_true', help='hide confidences')
    parser.add_argument('--half', action='store_true', help='use FP16 half-precision inference')
    parser.add_argument('--dnn', action='store_true', help='use OpenCV DNN for ONNX inference')
    opt = parser.parse_args()
    opt.imgsz *= 2 if len(opt.imgsz) == 1 else 1  # expand
    print_args(FILE.stem, opt)
    return opt


def main(opt):
    check_requirements(exclude=('tensorboard', 'thop'))
    run(**vars(opt))

def upload():
    pattern = "images\\*"
    files = glob.glob(pattern)

    # deleting the files with txt extension
    for file in files:
        os.remove(file)
    progress_upload['value']=0
    f1.update_idletasks()
    time.sleep(0.000005)
    pattern = "images\\*.jpg"
    files = glob.glob(pattern)
    global counter_11
    counter_11=0
# deleting the files with txt extension
    for file in files:
        os.remove(file)
    global counter
    counter=0
    global fln
    global namess
    fln=filedialog.askopenfilenames(initialdir=os.getcwd(),title="select images", filetypes=(("dcm files","*.dcm"),("all files","*.*")))
    var = root.tk.splitlist(fln)
    filePaths = []
    for f in var:
        filePaths.append(f)
    print(filePaths)
    destination_folder = r"images\\"
    for img_path in fln:
        counter_11=counter_11+1
        print(img_path)
        
        a=(100/len(fln))*counter_11
        progress_upload['value']=a
        f1.update_idletasks()
        b= "% s" % a
        #time.sleep(.0000005)
        lbl12.configure(text="Uploading :" + b+"%", font="bold")
        ds = dcmread(img_path) 
        M=ds.pixel_array 
        array = np.full((M.shape[0], M.shape[1]), 65535)
        array=array.astype(np.uint16)
        DAPI=array-M
        histogram, bin_edges = np.histogram(DAPI.ravel(), bins=range(65536))
        '''print(histogram.shape)
        x = (loaded_model.predict([histogram]))
        print(x)'''
        


        x=[]
        d=0
        for i in range(65534,0,-1):
            if histogram[i]>25000 and d<5 :
                x.append(i)
                d=d+1
            
                
        y=[]
        c=0
        for i in range(0,65535):
            if histogram[i]>50 and c<5:
                y.append(i)
                c=c+1
                
        z=[]
        d=0
        for i in range(65534,0,-1):
            if histogram[i]>250 and d<40 :
                z.append(i)
                d=d+1
        


            
        if len(x)==0:
            m=stretch2(DAPI,lower_thresh=y[len(y)-1]-5, upper_thresh=z[len(z)-1]+800)
        else:
             m=stretch2(DAPI,lower_thresh=y[len(y)-1]-5, upper_thresh=x[len(x)-1]+500)
        #y=stretch2(DAPI,lower_thresh=x[0], upper_thresh=x[len(x)-1])
        #z=y.astype(np.uint8)

    # Importing Image and ImageFilter module from PIL package 
          
        # applying the EDGE_ENHANCE filter

    # Importing Image and ImageFilter module from PIL package 
          
        # applying the EDGE_ENHANCE filter
        
        #pil = Image.fromarray(m)pip install torch
        pil1 = Image.fromarray(m.astype(np.uint8))
        #obj = ImageEnhance.Brightness(pil1)
        #pil1=obj.enhance(1)
        #img_shr_obj=ImageEnhance.Sharpness(pil1)
        #factor=1.5   # Specified Factor for Enhancing Sharpness
        #e_i=img_shr_obj.enhance(factor)
        #e_i = ImageOps.autocontrast(pil1, cutoff=10)
        #e_i = pil1.filter(ImageFilter.UnsharpMask(radius = 50, percent = 5000))
        #im2 = pil1.filter(ImageFilter.UnsharpMask())
        #e_i=e_i.transpose(Image.ROTATE_90)
        directory = os.path.basename(img_path)
  
        # Parent Directory path
        parent_dir = "images\\"
          
        # Path
        path = os.path.join(parent_dir, directory)
          
        # Create the directory
        # 'GeeksForGeeks' in
        # '/home / User / Documents'
        os.mkdir(path)
        pil1.save(path+"\\"+os.path.basename(img_path)+".png")
        print(a)
        image_bw=cv2.imread(path+"\\"+os.path.basename(img_path)+".png",0)
        image_bw=cv2.medianBlur(image_bw,5)
        # Apply 3x3 and 7x7 Gaussian blur
        clahe = cv2.createCLAHE(clipLimit = 5,tileGridSize=(32, 32))
        final_img = clahe.apply(image_bw)
        cv2.imwrite(path+"\\"+os.path.basename(img_path)+".png",final_img)
        
        '''file=os.path.join("images\\", os.path.basename(img_path))
        cv2.imwrite(file, crop)'''
    img2=Image.open("images\\"+os.path.basename(fln[counter])+".png")
    img2.thumbnail((450,337))
    img_2=ImageTk.PhotoImage(img2)
    
    lbl1.configure(image=img_2)
    lbl1.image=img_2
    str1=os.path.basename(fln[counter])
    T2.configure(text=str1, font= "bold")
    lbl17.configure(text= str(1), font ="bold")
    
    # fetch all files
def forward():
    global counter
    global all_defects
    global fln
    global namess
    if (counter<=len(fln)):
        img2=Image.open(os.path.join("images\\",namess[counter+1]))
        img2.thumbnail((450,337))
        img_2=ImageTk.PhotoImage(img2)
        lbl1.configure(image=img_2)
        lbl1.image=img_2
        img3=Image.open(os.path.join("runs/detect/exp",namess[counter+1]))
        img3.thumbnail((450,337))
        img_3=ImageTk.PhotoImage(img3)
        lbl0.configure(image=img_3)
        lbl0.image=img_3
        T2.configure(text= namess[counter+1], font ="bold")
        lbl17.configure(text= counter+2, font ="bold")
        if (all_defects[counter+1][0]== ''):
            T3.configure(text="Non-defective film", font="bold")
        else:
            T3.configure(text="Defective film" + "\n" + "AI-based prediction :"+opq[counter+1] + "\n" +"AI + Feature-based prediction :"+op[counter+1], font="bold")
        counter=counter+1
def backward():
    global counter
    global counter_2
    global all_defects
    global namess
    global fln
    counter=counter-1
    if (counter>-1):
        
        img_22=Image.open(os.path.join("images\\",namess[counter]))
        img_22.thumbnail((450,337))
        img__22=ImageTk.PhotoImage(img_22)
        lbl1.configure(image=img__22)
        lbl1.image=img__22
        img_33=Image.open(os.path.join("runs/detect/exp",namess[counter]))
        img_33.thumbnail((450,337))
        img__33=ImageTk.PhotoImage(img_33)
        lbl0.configure(image=img__33)
        lbl0.image=img__33
        T2.configure(text= namess[counter], font ="bold")
        lbl17.configure(text= counter+1, font ="bold")
        if (all_defects[counter][0]== ''):
            T3.configure(text="Non-defective film", font="bold")
        else:
            T3.configure(text="Defective film" + "\n" + "AI-based prediction :"+opq[counter] + "\n" +"AI + Feature-based prediction :"+op[counter], font="bold")
        
        
def detect():
    global counter_111
    counter_111=0
    progress_detect['value']=0
    f1.update_idletasks()
    time.sleep(0.000005)
    pattern = r"runs/detect/exp*"
    for item in glob.iglob(pattern, recursive=True):
        shutil.rmtree(item)
    opt=parse_opt()
    main(opt)
    global fln
    global counter_2
    global namess
    counter_2=0
    global all_defects
    global counter_1
    str1="Total number of images uploaded ="
    str2="Total number of defective samples ="
    str3="Total number of non-defective samples ="
    str4=str1+str(len(fln))
    str5=str2+str(counter_1)
    str6=str3+str((len(fln)-counter_1))
    T1.configure(text=str4 +"\n" +str5 + "\n" +str6,font='bold')
    img2=Image.open(os.path.join("runs/detect/exp",namess[counter_2]))
    img2.thumbnail((450,337))
    img_2=ImageTk.PhotoImage(img2)
    
    img3=Image.open(os.path.join("images\\",namess[counter_2]))
    img3.thumbnail((450,337))
    img_3=ImageTk.PhotoImage(img3)
    lbl0.configure(image=img_2)
    lbl0.image=img_2
    lbl1.configure(image=img_3)
    lbl1.image=img_3
    T2.configure(text=namess[counter_2], font="bold")
    if (all_defects[counter_2][0]== ''):
        T3.configure(text="Non-defective film", font="bold")
    else:
        T3.configure(text="Defective film" + "\n" + "AI-based prediction :"+opq[counter_2] + "\n" +"AI + Feature-based prediction :"+op[counter_2], font="bold")
    

    
root = Tk()
root.title('Check Look-a-like Product: Tata motors Ltd.')
root.geometry("1530x1530")
root.configure(bg="black")
notebook = ttk.Notebook(root)
notebook.pack(pady=15)
p0= PhotoImage(file='background.png')
f1 = Frame(notebook, width=1500, height=1500,bg="lightSkyBlue2")



f1.pack(fill="both", expand=1)

f1.pack(fill="both", expand=1)
p1= PhotoImage(file='logo.png')
p2= PhotoImage(file='upload.png')
#p3= PhotoImage(file='product1.png')
#p4= PhotoImage(file='product2.png')
p5= PhotoImage(file='detect.png')
p6= PhotoImage(file='PREV.png')
p7= PhotoImage(file='NEXT.png')
p8=PhotoImage(file='excel.png')
p9=PhotoImage(file='uploaded.png')
p10=PhotoImage(file='detections.png')
f2 = Frame(f1, width=110, height=350,bg="yellow")
f2.place(x=650,y=170)
f3 = Frame(f1, width=110, height=350,bg="yellow")
f3.place(x=1150,y=170)
lbl0=Label(f3,bg="yellow")
lbl0.place(x=0,y=0)
lbl1=Label(f2,bg="yellow")
lbl1.place(x=0,y=0)
lbl2=Label(f1,image=p1,bg="black")
lbl2.place(x=25,y=25)
lbl3=Button(f1,image=p2,bg="black",command=upload)
lbl3.place(x=25,y=250)
#lbl4=Button(f1,image=p3,bg="black")
#lbl4.place(x=25,y=330)
#lbl5=Button(f1,image=p4,bg="black")
#lbl5.place(x=250,y=330)
lbl6=Button(f1,image=p5,bg="black",command=detect)
lbl6.place(x=240,y=250)
#lbl7=Button(f1,image=p6,bg="black")
#lbl7.place(x=25,y=460)
#lbl8=Label(f1,image=p7,bg="black")
#lbl8.place(x=245,y=460)
T1 = Label(f1,bg="lightSkyBlue2",height=8)
T1.place( x=25,y=520,width=400)

T2 = Label(f1,bg="lightSkyBlue2",height=3)
T2.place( x=750,y=580,width=500)
T3 = Label(f1,bg="lightSkyBlue2",height=5)
T3.place( x=750,y=650,width=500)
lbl9=Button(f1,image=p6,bg="black",command=backward)
lbl9.place(x=600,y=620)
lbl10=Button(f1,image=p7,bg="black", command=forward)
lbl10.place(x=1300,y=620)

progress_upload = ttk.Progressbar(f1, orient = HORIZONTAL,
              length = 400, mode = 'determinate')
progress_upload.place(x=25,y=320)
progress_detect = ttk.Progressbar(f1, orient = HORIZONTAL,
              length = 400, mode = 'determinate')
progress_detect.place(x=25,y=420)


lbl11=Button(f1,image=p8,bg="black", command=excel)
lbl11.place(x=1220,y=25)

lbl12=Label(f1,bg="lightSkyBlue2",height=2)
lbl12.place(x=25,y=350,width=400)
lbl13=Label(f1,bg="lightSkyBlue2",height=2)
lbl13.place(x=25,y=450,width=400)

lbl14=Label(f1,bg="lightSkyBlue2",height=2)
lbl14.place(x=1050,y=80,width=400)

lbl15=Label(f1,image=p9)
lbl15.place(x=630,y=120)

lbl16=Label(f1,image=p10)
lbl16.place(x=1130,y=120)

lbl17=Label(f1,bg="lightSkyBlue2",height=2)
lbl17.place(x=950,y=550,width=40)

root.mainloop()
