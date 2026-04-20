import os
import argparse
import rasterio
import cv2

def make_patches(im_dir, out_dir, patch_size=512, stride=200, out_format='tif', prefix='train_'):

    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
        
    # image_dataset = []  
    for path, subdirs, files in os.walk(im_dir):
        subdirs.sort()
        dirname = path.split(os.path.sep)[-1]
        images = sorted(os.listdir(path))  #List of all image names in this subdirectory
        
        index = 0
        for image_name in images:
            if image_name.endswith(".jpg") or \
				image_name.endswith(".png") or \
				image_name.endswith(".tif"):   # Only read jpg images...

                with rasterio.open(os.path.join(path, image_name)) as src:
                    # image = cv2.imread(os.path.join(path, image_name), 1)  # Read each image as BGR
                    image = src.read().transpose(1, 2, 0)
                    profile = src.profile
                    # print(os.path.join(path, image_name))
                    # print(image)
                    print(image.shape)
                
                '''
                如果 x + patch_size > image.shape[1]，则 x = image.shape[1] - patch_size
                x = image.shape[1] - patch_size 这句话的意思是，在最后一个patch的时候通过向后重叠的方式来生成patch，以保证不会有遗漏
                '''
                for x in range(0, image.shape[0], stride):
                    for y in range(0, image.shape[1], stride):
                        
                        # im_name = image_name[:-4] + '_' + str(x) + '_' + str(y) + '.'+ out_format
                        # print(im_name)

                        if (y + patch_size > image.shape[0]): # check if x is out of bound
                            y = image.shape[0] - patch_size
                        if (x + patch_size > image.shape[1]): # check if y is out of bounds
                            x = image.shape[1] - patch_size
                        tile = image[y : y+patch_size, x : x+patch_size]

                        im_name = prefix + str(index) + '.'+ out_format
                        index = index + 1
                        print(im_name)

                        # with rasterio.open(os.path.join(out_dir, im_name), 'w', **profile) as dst:
                        #     dst.write(tile)
                        cv2.imwrite(os.path.join(out_dir, im_name), tile)
                # print("tile", tile.shape)
                # break

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--in_dir', type=str, default=None, help='directory containing images to be patchified', required=True)
    parser.add_argument('--out_dir', type=str, default=None, help='directory to save the patches to', required=True)
    parser.add_argument('--patch_size', type=int, default=512, help='size of the square patches to be created in pixels per side') # 256
    parser.add_argument('--stride', type=int, default=400, help='number of pixels to move the window creating the patches') # 400
    parser.add_argument('--output_format', choices=['png', 'tif', 'jpg'], default='tif')
    opt = parser.parse_args()

    make_patches(opt.in_dir, opt.out_dir, opt.patch_size, opt.stride, out_format=opt.output_format, prefix='val_')



# ls -l | grep "^-" | wc -l
# ls -lR | grep "^-" | wc -l

# python patchify_ISPRS.py --in_dir /home/icclab/Documents/lqw/DatasetMMF/Vaihingen_Orgin/Val_DSMCopy --out_dir /home/icclab/Documents/lqw/DatasetMMF/Vaihingen/val/DSM256 --patch_size 480 --stride 240
# python patchify_ISPRS.py --in_dir /home/icclab/Documents/lqw/DatasetMMF/Vaihingen_Orgin/val_imagesCopy --out_dir /home/icclab/Documents/lqw/DatasetMMF/VaihingenRef/images --patch_size 480 --stride 240
# python patchify_ISPRS.py --in_dir /home/icclab/Documents/lqw/DatasetMMF/Vaihingen_Orgin/val_masksCopy --out_dir /home/icclab/Documents/lqw/DatasetMMF/VaihingenRef/masks --patch_size 480 --stride 240


# python patchify_ISPRS.py --in_dir /home/icclab/Documents/lqw/DatasetMMF/Vaihingen_Orgin/test_DSMCopy --out_dir /home/icclab/Documents/lqw/DatasetMMF/Vaihingen/test/DSM256 --patch_size 480 --stride 240
# python patchify_ISPRS.py --in_dir /home/icclab/Documents/lqw/DatasetMMF/Vaihingen_Orgin/test_imagesCopy --out_dir /home/icclab/Documents/lqw/DatasetMMF/VaihingenRef/images --patch_size 480 --stride 240
# python patchify_ISPRS.py --in_dir /home/icclab/Documents/lqw/DatasetMMF/Vaihingen_Orgin/test_masksCopy --out_dir /home/icclab/Documents/lqw/DatasetMMF/VaihingenRef/masks --patch_size 480 --stride 240

# python patchify_ISPRS.py --in_dir /home/icclab/Documents/lqw/DatasetMMF/Vaihingen_Orgin/Train_DSMCopy --out_dir /home/icclab/Documents/lqw/DatasetMMF/Vaihingen/train/DSM256 --patch_size 480 --stride 240
# python patchify_ISPRS.py --in_dir /home/icclab/Documents/lqw/DatasetMMF/Vaihingen_Orgin/train_imagesCopy --out_dir /home/icclab/Documents/lqw/DatasetMMF/temp/images --patch_size 480 --stride 240
# python patchify_ISPRS.py --in_dir /home/icclab/Documents/lqw/DatasetMMF/Vaihingen_Orgin/train_masksCopy --out_dir /home/icclab/Documents/lqw/DatasetMMF/temp/masks --patch_size 480 --stride 240


# test
# python patchify.py --in_dir /home/icclab/Documents/lqw/DatasetMMF/Vaihingen_Orgin/test_DSMCopy --out_dir /home/icclab/Documents/lqw/DatasetMMF/Vaihingen/DSM256 --patch_size 256 --stride 200
# python patchify.py --in_dir /home/icclab/Documents/lqw/DatasetMMF/Vaihingen_Orgin/test_imagesCopy --out_dir /home/icclab/Documents/lqw/DatasetMMF/Vaihingen/images256 --patch_size 256 --stride 200
# python patchify.py --in_dir /home/icclab/Documents/lqw/DatasetMMF/Vaihingen_Orgin/test_masksCopy --out_dir /home/icclab/Documents/lqw/DatasetMMF/Vaihingen/masks256 --patch_size 256 --stride 200


# train
# python patchify.py --in_dir /home/icclab/Documents/lqw/DatasetMMF/Vaihingen_Orgin/Train_DSMCopy --out_dir /home/icclab/Documents/lqw/DatasetMMF/Vaihingen/DSM256 --patch_size 256 --stride 200
# python patchify.py --in_dir /home/icclab/Documents/lqw/DatasetMMF/Vaihingen_Orgin/train_imagesCopy --out_dir /home/icclab/Documents/lqw/DatasetMMF/Vaihingen/images256 --patch_size 256 --stride 200
# python patchify.py --in_dir /home/icclab/Documents/lqw/DatasetMMF/Vaihingen_Orgin/train_masksCopy --out_dir /home/icclab/Documents/lqw/DatasetMMF/Vaihingen/masks256 --patch_size 256 --stride 200


# val
# python patchify.py --in_dir /home/icclab/Documents/lqw/DatasetMMF/Vaihingen_Orgin/Val_DSMCopy --out_dir /home/icclab/Documents/lqw/DatasetMMF/Vaihingen/DSM256 --patch_size 256 --stride 200
# python patchify.py --in_dir /home/icclab/Documents/lqw/DatasetMMF/Vaihingen_Orgin/val_imagesCopy --out_dir /home/icclab/Documents/lqw/DatasetMMF/Vaihingen/images256 --patch_size 256 --stride 200
# python patchify.py --in_dir /home/icclab/Documents/lqw/DatasetMMF/Vaihingen_Orgin/val_masksCopy --out_dir /home/icclab/Documents/lqw/DatasetMMF/Vaihingen/masks256 --patch_size 256 --stride 200






#############################################################################################################################################################################






# python patchify.py --in_dir /home/icclab/Documents/lqw/DatasetMMF/Potsdam_Orgin/test_DSMCopy --out_dir /home/icclab/Documents/lqw/DatasetMMF/Potsdam/test/DSM256 --patch_size 512 --stride 400
# python patchify.py --in_dir /home/icclab/Documents/lqw/DatasetMMF/Potsdam_Orgin/test_masksCopy --out_dir /home/icclab/Documents/lqw/DatasetMMF/Potsdam/test/masks256 --patch_size 512 --stride 400
# python patchify.py --in_dir /home/icclab/Documents/lqw/DatasetMMF/Potsdam_Orgin/test_RGBCopy --out_dir /home/icclab/Documents/lqw/DatasetMMF/Potsdam/test/images256 --patch_size 512 --stride 400

# python patchify.py --in_dir /home/icclab/Documents/lqw/DatasetMMF/Potsdam_Orgin/train_DSMCopy --out_dir /home/icclab/Documents/lqw/DatasetMMF/Potsdam/train/DSM256 --patch_size 512 --stride 400
# python patchify.py --in_dir /home/icclab/Documents/lqw/DatasetMMF/Potsdam_Orgin/train_masksCopy --out_dir /home/icclab/Documents/lqw/DatasetMMF/Potsdam/train/masks256 --patch_size 512 --stride 400
# python patchify.py --in_dir /home/icclab/Documents/lqw/DatasetMMF/Potsdam_Orgin/train_RGBCopy --out_dir /home/icclab/Documents/lqw/DatasetMMF/Potsdam/train/images256 --patch_size 512 --stride 400

# python patchify.py --in_dir /home/icclab/Documents/lqw/DatasetMMF/Potsdam_Orgin/val_DSMCopy --out_dir /home/icclab/Documents/lqw/DatasetMMF/Potsdam/val/DSM256 --patch_size 512 --stride 400
# python patchify.py --in_dir /home/icclab/Documents/lqw/DatasetMMF/Potsdam_Orgin/val_masksCopy --out_dir /home/icclab/Documents/lqw/DatasetMMF/Potsdam/val/masks256 --patch_size 512 --stride 400
# python patchify.py --in_dir /home/icclab/Documents/lqw/DatasetMMF/Potsdam_Orgin/val_RGBCopy --out_dir /home/icclab/Documents/lqw/DatasetMMF/Potsdam/val/images256 --patch_size 512 --stride 400





# test
# python patchify.py --in_dir /home/icclab/Documents/lqw/DatasetMMF/Potsdam_Orgin/test_DSMCopy --out_dir /home/icclab/Documents/lqw/DatasetMMF/Potsdam/DSM256 --patch_size 512 --stride 400
# python patchify.py --in_dir /home/icclab/Documents/lqw/DatasetMMF/Potsdam_Orgin/test_RGBCopy --out_dir /home/icclab/Documents/lqw/DatasetMMF/Potsdam/images256 --patch_size 512 --stride 400
# python patchify.py --in_dir /home/icclab/Documents/lqw/DatasetMMF/Potsdam_Orgin/test_masksCopy --out_dir /home/icclab/Documents/lqw/DatasetMMF/Potsdam/masks256 --patch_size 512 --stride 400


# train
# python patchify.py --in_dir /home/icclab/Documents/lqw/DatasetMMF/Potsdam_Orgin/train_DSMCopy --out_dir /home/icclab/Documents/lqw/DatasetMMF/Potsdam/DSM256 --patch_size 512 --stride 400
# python patchify.py --in_dir /home/icclab/Documents/lqw/DatasetMMF/Potsdam_Orgin/train_RGBCopy --out_dir /home/icclab/Documents/lqw/DatasetMMF/Potsdam/images256 --patch_size 512 --stride 400
# python patchify.py --in_dir /home/icclab/Documents/lqw/DatasetMMF/Potsdam_Orgin/train_masksCopy --out_dir /home/icclab/Documents/lqw/DatasetMMF/Potsdam/masks256 --patch_size 512 --stride 400


# val
# python patchify.py --in_dir /home/icclab/Documents/lqw/DatasetMMF/Potsdam_Orgin/val_DSMCopy --out_dir /home/icclab/Documents/lqw/DatasetMMF/Potsdam/DSM256 --patch_size 512 --stride 400
# python patchify.py --in_dir /home/icclab/Documents/lqw/DatasetMMF/Potsdam_Orgin/val_RGBCopy --out_dir /home/icclab/Documents/lqw/DatasetMMF/Potsdam/images256 --patch_size 512 --stride 400
# python patchify.py --in_dir /home/icclab/Documents/lqw/DatasetMMF/Potsdam_Orgin/val_masksCopy --out_dir /home/icclab/Documents/lqw/DatasetMMF/Potsdam/masks256 --patch_size 512 --stride 400
