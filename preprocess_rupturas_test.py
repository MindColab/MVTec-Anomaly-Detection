import cv2
import os
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from PIL import Image
import argparse
from glob import glob

# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--input_path", type=str,
    help="path to input folder cvat segmentation files", required=True)
ap.add_argument("-o", "--output_path", type=str,
    help="path to output dir", required=True)
args = vars(ap.parse_args())
#args = ap.parse_args()

def add_sqr_borders(cropped_image, cropped_mask):
    """
    Agrega bordes a las imágenes cropped_image y cropped_mask para obtener parches cuadrados.
    Si la imagen recortada original es más ancha que alta, se agregarán bordes arriba y abajo a ambas imágenes.
    Si la imagen recortada original es más alta que ancha, se agregarán bordes a la izquierda y derecha a ambas imágenes.

    Parámetros:
    - cropped_image: numpy array, imagen recortada del parche.
    - cropped_mask: numpy array, máscara del parche recortada.

    Salidas:
    - cropped_image: numpy array, imagen del parche recortada y con bordes agregados para obtener un parche cuadrado.
    - cropped_mask: numpy array, máscara del parche recortada y con bordes agregados para obtener un parche cuadrado.
    """
    # Calcula el tamaño máximo de los bordes a agregar
    border_size = max(cropped_image.shape[0], cropped_image.shape[1]) - \
    min(cropped_image.shape[0], cropped_image.shape[1])

    if border_size % 2 == 0:
        correction = 0
    else:
        correction = 1

    if cropped_image.shape[0] < cropped_image.shape[1]:
        # Agrega bordes arriba y abajo a la imagen recortada para obtener un parche cuadrado
        cropped_image = cv2.copyMakeBorder(cropped_image, border_size//2, border_size//2 + correction, 0, 0, \
                                           cv2.BORDER_CONSTANT, value=0)

        # Agrega bordes arriba y abajo a la máscara del parche para obtener un parche cuadrado
        cropped_mask = cv2.copyMakeBorder(cropped_mask, border_size//2, border_size//2 + correction, 0, 0, \
                                          cv2.BORDER_CONSTANT, value=0)
    else:
        # Agrega bordes arriba y abajo a la imagen recortada para obtener un parche cuadrado
        cropped_image = cv2.copyMakeBorder(cropped_image, 0, 0, border_size//2, border_size//2 + correction, \
                                           cv2.BORDER_CONSTANT, value=0)

        # Agrega bordes arriba y abajo a la máscara del parche para obtener un parche cuadrado
        cropped_mask = cv2.copyMakeBorder(cropped_mask, 0, 0, border_size//2, border_size//2 + correction, \
                                          cv2.BORDER_CONSTANT, value=0)


    return cropped_image, cropped_mask




def get_patch_mask(image, iris_mask, ruptura_mask, boder_size=5):
    """
    Genera un parche y una máscara del parche a partir de una imagen y una máscara.
    El parche se genera recortando la imagen y la máscara al bounding box de la máscara y agregando un borde al parche.
    La máscara del parche se genera a partir de la máscara recortada y el borde agregado al parche.

    Parámetros:
    - image_path: imagen.
    - mask_path: máscara.
    - border_size: int, tamaño del borde a agregar al parche.

    Salidas:
    - cropped_image: numpy array, imagen del parche recortada y con borde agregado.
    - patch_mask: numpy array, máscara del parche.
    """
    # Load the image and the mask
    #image = cv2.imread(image_path)
    #mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

    # Get the indices of the non-zero pixels in the mask
    non_zero_indices = cv2.findNonZero(iris_mask)

    # Calculate the bounding rectangle of the non-zero pixels
    x, y, w, h = cv2.boundingRect(non_zero_indices)

    # Crop the image to the bounding rectangle
    cropped_image = image[y:y+h, x:x+w]

    # Crop the mask to the bounding rectangle
    cropped_mask = ruptura_mask[y:y+h, x:x+w]

    # Crop the iris mask to the bounding rectangle
    cropped_iris_mask = iris_mask[y:y+h, x:x+w]

    # Add a 5-pixel border to the patch
    border_size = 5
    cropped_image = cv2.copyMakeBorder(cropped_image, border_size, border_size, border_size, border_size, \
                                       cv2.BORDER_CONSTANT, value=0)
    cropped_mask = cv2.copyMakeBorder(cropped_mask, border_size, border_size, border_size, border_size, \
                                      cv2.BORDER_CONSTANT, value=0)
    cropped_iris_mask = cv2.copyMakeBorder(cropped_iris_mask, border_size, border_size, border_size, border_size, \
                                      cv2.BORDER_CONSTANT, value=0)

    # Create the mask for the patch
    patch_mask = np.zeros((cropped_mask.shape[0], cropped_mask.shape[1]), dtype=np.uint8)
    patch_mask[border_size:-border_size, border_size:-border_size] = 1

    # Apply the mask to the patch mask
    patch_mask = cv2.bitwise_and(patch_mask, patch_mask, mask=cropped_mask)

    # Apply the mask to the image patch
    cropped_image = cv2.bitwise_and(cropped_image, cropped_image, mask=cropped_iris_mask)

    return cropped_image, patch_mask

def read_images_name(images_list_path):
    file1 = open(images_list_path, 'r')
    Lines = file1.readlines()
    # Strips the newline character
    images_list = []
    for line in Lines:
        images_list.append(line)
    return images_list

def read_labels(labels_path):
    file1 = open(labels_path, 'r')
    Lines = file1.readlines()
    # Strips the newline character
    label_dict = {}
    for line in Lines:
        if not line.startswith("#"):
            label = line.split(":")[0]
            color = line.split(":")[1]
            label_dict[label]= list(map(int,color.split(",")))

    print(label_dict)
    return label_dict

# extract cropping mask
def get_iris_mask(mask):
    print("mask_size:",mask.shape)
    img = np.zeros(mask.shape,dtype=np.uint8)
    img.fill(255) # or img[:] = 255
    result_mask= cv2.inRange(mask,np.array([1,1,1]),np.array([255,255,255]))
    masked = cv2.bitwise_and(img,img, mask=result_mask)
    masked = cv2.cvtColor(masked, cv2.COLOR_BGR2RGB)
    return masked

# apply mask to image
def apply_mask_to_image(image,mask):
    mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
    print("mask_size:",mask.shape)
    print("image_size:",image.shape)
    masked = cv2.bitwise_and(image,image, mask=mask)
    #masked = cv2.cvtColor(masked, cv2.COLOR_BGR2RGB)

    return masked

# extract masks by color
def extract_mask_by_color(mask,color):
    print("mask_size:",mask.shape)
    img = np.zeros(mask.shape,dtype=np.uint8)
    img.fill(255) # or img[:] = 255
    result_mask= cv2.inRange(mask,color,color)
    masked = cv2.bitwise_and(img,img, mask=result_mask)
    masked = cv2.cvtColor(masked, cv2.COLOR_BGR2RGB)
    return masked

# cropping by windows
def crop_window(image,mask,window_size):
    mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
    contours,_ = cv2.findContours(mask.copy(), 1, 1) # not copying here will throw an error
    x,y,w,h = cv2.boundingRect(contours[0])
    corner_x = abs(int((x+w/2) - (window_size/2)))
    corner_y = abs(int((y+h/2) - (window_size/2)))
    #print("corner_x",corner_x)
    #print("corner_y",corner_y)
    cropped_image = image[corner_y:corner_y+window_size, corner_x:corner_x+window_size]
    return cropped_image


def run():
    #print("args",args)
    #print(args['input_path'])
    #Output csv
    csv_out = list()
    out_csv_path = os.path.dirname(args['output_path'])+"/test-patches.csv"
    # Inicializa la dimensión mínima en un valor muy grande
    min_dimension = float('inf')

    imageid=0

    inference_folders = glob(f"{args['input_path']}/*/", recursive = False)
    for folder in inference_folders:
        #if not "task228" in folder:
        #    continue
        print(f"################### processing: {folder}")
        # Variables
        # Set input:
        #INPUT_DIR = "../dataset/inferencia/task224/"
        #OUPUT_DIR = "../outdataset/task224/"
        INPUT_DIR = folder
        OUTPUT_DIR = args["output_path"] #+ folder.split("/")[-2]

        print("input:",INPUT_DIR)
        print("output:",OUTPUT_DIR)
        #assert j


        CHECK_MASK_EXISTS = True

        images_list_path = f"{INPUT_DIR}/ImageSets/Segmentation/default.txt"
        labels_path = f"{INPUT_DIR}/labelmap.txt"
        images_path = f"{INPUT_DIR}/JPEGImages/"
        masks_path = f"{INPUT_DIR}/SegmentationClass/"
        img_ext = ".PNG"
        mask_ext = ".png"
        out_mask_path = f"{OUTPUT_DIR}/mask/"
        out_image_path = f"{OUTPUT_DIR}/rupturas/"


        #read labels
        label_dict = read_labels(labels_path)

        #Create output dir
        os.makedirs(out_image_path, exist_ok=True)
        os.makedirs(out_mask_path, exist_ok=True)



        for img in read_images_name(images_list_path):
            image_name = img.strip()
            image_path = f"{images_path}{image_name}{img_ext.lower()}"
            mask_path = f"{masks_path}{image_name}{mask_ext.lower()}"
            prefix_image = folder.split("/")[-2]

            if not os.path.exists(image_path):
                image_path = f"{images_path}{image_name}{img_ext.upper()}"

            if not os.path.exists(mask_path):
                image_path = f"{masks_path}{image_name}{mask_ext.upper()}"

            #print("in:",image_path)
            #print("in:",mask_path)

            # check if mask exists
            if os.path.exists(mask_path):
                print("in:",image_path)
                print("in:",mask_path)
                image = cv2.imread(image_path)
                #image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                if image.shape[0] > image.shape[1]:
                    dim = (1080,1920)
                else:
                    dim = (1920,1080)
                mask = cv2.imread(mask_path)
                mask = cv2.cvtColor(mask, cv2.COLOR_BGR2RGB)

                # resize
                #image = cv2.resize(image,dim)
                #mask = cv2.resize(mask,dim)
                ruptura_mask = extract_mask_by_color(mask,np.array(label_dict['ruptura']))
                ruptura_mask = cv2.cvtColor(ruptura_mask, cv2.COLOR_BGR2GRAY)
                iris_mask = get_iris_mask(mask)
                iris_mask = cv2.cvtColor(iris_mask, cv2.COLOR_BGR2GRAY)
                image, mask = get_patch_mask(image, iris_mask, ruptura_mask)
                cropped_image, cropped_mask = add_sqr_borders(image, mask)

                #cv2.namedWindow("output", cv2.WINDOW_NORMAL)
                #cv2.imshow("output", cropped_mask)
                #cv2.waitKey(0)
                #cv2.destroyAllWindows()

                #cropped_image = cv2.cvtColor(cropped_image, cv2.COLOR_RGB2BGR)
                #cropped_mask = cv2.cvtColor(cropped_mask, cv2.COLOR_RGB2BGR)

                # Actualiza la dimensión mínima con la menor dimensión de la imagen actual
                min_dimension = min(min_dimension, min(cropped_image.shape[0], cropped_image.shape[1]))

                # check if segmentation exists
                im_pil = Image.fromarray(cropped_mask)
                if CHECK_MASK_EXISTS == False or not im_pil.getbbox() == None:
                    # save images and masks
                    image_basename = os.path.basename(image_name)
                    image_out_path = f"{out_image_path}{imageid}_{prefix_image}_{image_basename}{img_ext.lower()}"
                    mask_out_path = f"{out_mask_path}{imageid}_mask_{prefix_image}_{image_basename}_{imageid}{mask_ext.lower()}"
                    print("out1:",image_out_path)
                    print("out2:",mask_out_path)
                    cv2.imwrite(image_out_path, cropped_image)
                    cropped_mask = cv2.normalize(cropped_mask, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8UC1)
                    cv2.imwrite(mask_out_path, cropped_mask)

                    data = {}
                    data['image'] = f"{imageid}_{prefix_image}_{image_basename}{img_ext.lower()}"
                    data['mask'] = f"{imageid}_mask_{prefix_image}_{image_basename}{img_ext.lower()}"
                    data['class'] = "ruptura"
                    csv_out.append(data)
                    imageid = imageid + 1

    print(f"Dimension minima detectada:{min_dimension}x{min_dimension}")

    # Recorre recursivamente el directorio y subdirectorios y redimensiona al menor tamaño de las imagenes

    #force min_dimension to 256
    min_dimension = 256

    print(f"Redimensionando imagenes a {min_dimension}x{min_dimension}")
    for root, dirs, files in os.walk(args["output_path"] + "/"):
        # Por cada archivo de imagen en el directorio actual
        for file in files:
            # Si es un archivo de imagen
            if file.endswith('.jpg') or file.endswith('.png'):
                # Lee la imagen
                image = cv2.imread(os.path.join(root, file))
                # Redimensiona la imagen al tamaño mínimo
                image = cv2.resize(image, (min_dimension, min_dimension))
                # Guarda la imagen redimensionada
                cv2.imwrite(os.path.join(root, file), image)
    # save csv
    csv = pd.DataFrame(csv_out)
    csv.to_csv(out_csv_path, index=False)

run()

'''
    # Read color image
    image = cv2.imread(image_path)
    image2 = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) #Converts from one colour space to the other
    iris_mask = get_iris_mask(mask2)
'''
