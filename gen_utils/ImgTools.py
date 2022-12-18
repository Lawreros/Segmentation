
from typing import List, Iterable, Tuple, Any, Union, Generator, Dict
import numpy as np
from PIL import Image
import os
import nibabel as nib
from pathlib import Path
import pydicom
import cv2
from skimage.transform import rotate, AffineTransform, warp, rescale, resize
import math


class ImgIE():
    '''Class for the loading of images into numpy arrays and the saving of numpy arrays into images.
    Also handles rudimentary processing of the images.'''
    def __init__(self) -> None:
        '''Test this out'''
        pass

    def load_image(self, im_path: Path, unit: str='intensity', verbose: bool=False) -> np.ndarray:
        # Given an image path, determines the function required to load the contents
        # as a numpy array, which is returned.
        fil_typ = os.path.splitext(im_path)[1]

        if fil_typ == '.png':
            # If file is a png
            with Image.open(im_path) as f_im:
                img = np.array(f_im)

            if verbose:
                print(f'Loading {im_path} as png')
                print(f'Image shape:{img.shape}')

            if unit == 'intensity':
                if len(img.shape)==3:
                    img = self.rgba2ycbcr(img)
                    img = img[:,:,0] #Just deal with intensity values at the moment because 
                                    # having multiple channels throws off cv2 when saving, 
                                    # since it also does BGR instead of RGB and will save a blue image
                elif len(img.shape)==2:
                    pass            # If the png is just greyscale, then there is nothing that can
                                    # be done except take the single channel
                else:
                    raise ImportError("Provided png image is not 2 or 3 dimensional, something is wrong with the image.")

            elif unit == 'color':
                raise NotImplementedError("""Loading and creation of patches from color png images is
                currently not supported. Please use template['unit']='intensity' for conversion of png
                imges to greyscale intesity images.""")

        elif fil_typ == '.nii' or fil_typ == '.gz':
            img = nib.load(im_path).get_fdata()
            if verbose:
                print(f'Loading {im_path} as nii')
                print(f'Image shape:{img.shape}')

        elif fil_typ == '.dcm':
            img = pydicom.dcmread(im_path).pixel_array
            if verbose:
                print(f'Loading {im_path} as dicom')
                print(f'Image shape:{img.shape}')

        else:
            raise FileNotFoundError(f'Image file type {fil_typ} not supported.')

        return img


    def load_png(self, im_path: Path, unit: str='raw') -> np.ndarray:
        '''Load png image
        
        Parameters
        ----------
        im_path : Path
            Path to the file you wish to load.
        
        unit : str
            What the unit of the given image should be. Current options are:
            - `'raw'` : the raw RGBA values stored in the image (Default)
            - `'lumanince'` :  the intensity channel from converting the RGB image to YCbCr.
            This will result in the image only having one channel

        Returns
        -------
        img : float ndarray
            The loaded image as a numpy array

        '''
        with Image.open(im_path) as f_im:
            if unit == 'raw':
                return np.array(f_im)
            if unit == 'luminance':
                img = np.array(f_im)
                
                if len(img.shape) == 3:
                    # convert to YCbCr then take first channel (luminance)
                    return self.rgba2ycbcr(img)[:,:,0]
                
                elif len(img.shape) == 2:
                    return img
                
                else:
                    raise ImportError("Provided png image is not 2 or 3 dimensional, something is wrong with the image.")
        
            # If the unit type is not supported
            raise NotImplementedError(f'Loading of png using unit value {unit} is currently not supported.')


    def load_jpg(self, im_path: Path, unit: str='raw') -> np.ndarray:
        '''Load jpg image

        '''
        pass

    def load_nifti(self, im_path: Path) -> np.ndarray:
        '''Load nifti file from provided path

        Parameters
        ----------
        im_path : Path
            Path to the file you wish to load.
        
        Returns
        -------
        img : float ndarray
            The loaded image as a numpy array
        
        '''
        return nib.load(im_path).get_fdata()

    def load_dicom(self, im_path: Path) -> np.ndarray:
        '''Load DICOM file from provided path

        Parameters
        ----------
        im_path : Path
            Path to the file you wish to load.
        
        Returns
        -------
        img : float ndarray
            The loaded image as a numpy array
        
        '''
        return pydicom.dcmread(im_path).pixel_array

    def rgba2ycbcr(self, img_rgba: np.ndarray) -> np.ndarray:
        '''Takes an RBG image and returns it as a YCbCr image 

        Parameters:
        ----------
        img_rgb : ndarray
            The RGBA image which you want to convert to YCbCr

        Returns:
        --------
        img_ycbcr : float ndarray
            The converted image

        '''
        if len(img_rgba.shape) != 4:
            raise ValueError('Input image is not RGBA')

        img_rgb = img_rgba.astype(np.float32)
        
        img_ycrcb = cv2.cvtColor(img_rgba, cv2.COLOR_RGB2YCR_CB)
        img_ycbcr = img_ycrcb[:,:,(0,2,1)].astype(np.float32)
        img_ycbcr[:,:,0] = (img_ycbcr[:,:,0]*(235-16)+16)/255.0
        img_ycbcr[:,:,1:] = (img_ycbcr[:,:,1:]*(240-16)+16)/255.0

        return img_ycbcr

    
    def save_image(self, im: np.ndarray, fname: Path, verbose: bool = False) -> Path:
        # Take a given image and save it as the specified format. The use of a Path variable type
        # is intentional, as there should be less chance of incorrect entries
        # fname = output name of the saved file, including the suffix
        # im = numpy array of image
        #
        # returns fname, so it can be easily appended to a path if necessary

        dim = im.shape #Get number of dimensions of image

        if fname.suffix == '.png':
            # Check that you aren't saving a 3D image
            #TODO: Scale inputs to [0,255] so data isn't lost/image isn't saturated
            cv2.imwrite(f'{fname}',im)
            if verbose:
                print(f'Saving: {fname}')
        elif fname.suffix == '.nii' or fname.suffix == '.gz':

            # TODO: Add option to transpose image for some reason because mricron hates the first dim[0] = 1
            # Still gets loaded fine in terms of loading into python, but visualizing it is bad
            # np.transpose(im, (1,2,0))


            # TODO: If image is 2D then append a third  dimension before saving(?)
            if len(dim) < 4:
                nib.save(nib.Nifti1Image(im, np.eye(len(dim)+1)), fname)
            else:
                #nibabel can only support up to 4-by-4 affines
                nib.save(nib.Nifti1Image(im, np.eye(4)), fname)
            if verbose:
                print(f'Saving: {fname}')
        elif fname.suffix == '.dcm':
            raise NotImplementedError('DICOM saving currently not supported')
        else:
            raise NotImplementedError(f'Specified file type {fname.suffix} for {fname} is currently not supported for saving')

        return fname





from skimage.transform import rotate, AffineTransform, warp, rescale, resize
import math

class ImgAug(ImgIE):
    def __init__(self) -> None:
        super(ImgIE, self).__init__()
        # super().__init__()
        # pass


    def gen_random_aug(self, params: Dict[str,List[int]], float_params: bool=False, negative=True) -> Generator[Dict[str,List[int]],None, None]:
        '''Generator which takes a dictionary of boundary vaules and returns randomly selected values within those boundaries.
        Used for the creation of randomized augmentation parameters for other methods in this class.
        
        Parameters:
        ----------
        params : dict[str, list[int]]
            The boundaries to be used in random augmentation parameter generation
        float_params: bool
            Whether the returned randomly selected values can be floats, instead of integers. Default is False
        negative: bool
            Whether the randomly generated values can also be negative. Thus an input value of [4] can return any value from [-4, 4].
            Default is True

        Returns
        -------
        out_params : dict[str, list[int or float]]
            The randomly generated values for the input dictionary

        '''

        while True:
            out_params = {}
            if float_params:
                for ky, i in params.items():
                    out_params[ky] = [np.random.uniform(-k*negative, k) if k!=0 else k for k in i]
            else:
                for ky, i in params.items():
                    out_params[ky] = [np.random.randint(-k*negative,k) if k!=0 else k for k in i]

            yield out_params


    def array_translate(self, im: np.ndarray, trans: List[int], mode: str='symmetric') -> "tuple[np.ndarray, str]":
        # Translation
        dim = im.shape

        if len(dim) != len(trans):
            raise IndexError(f'Translation of numpy array with dimensions: {dim} is not compatible with translation {trans}')
        
        if len(dim) == 2:
            transform = AffineTransform(translation=(trans[0], trans[1]))
            im = warp(im, transform, mode=mode)
            label = f'_tr{trans[0]}_{trans[1]}'

        elif len(trans) == 3:
            transform = AffineTransform(translation=(trans[1], trans[2]))
            for i in range(dim[0]):
                im[i,:,:] = warp(im[i,:,:], transform, mode = mode)

            for i in range(dim[1]):
                # Because two dimensions were already translated, you only need to translate
                # along one dimension
                im[:,i,:] = warp(im[:,i,:], AffineTransform(translation=(trans[0],0)), mode='symmetric')
                
            label = f'_tr{trans[0]}_{trans[1]}_{trans[2]}'

        elif len(trans) == 4:
            transform = AffineTransform(translation=(trans[1], trans[2]))
            for j in range(dim[3]):
                for i in range(dim[0]):
                    im[i,:,:,j] = warp(im[i,:,:,j], transform, mode = mode)

            for j in range(dim[2]):
                for i in range(dim[1]):
                # Because two dimensions were already translated, you can do the same thing to translate the remaining two
                # along one dimension
                    im[:,i,j,:] = warp(im[:,i,j,:], AffineTransform(translation=(trans[0],trans[3])), mode='symmetric')
                
            label = f'_tr{trans[0]}_{trans[1]}_{trans[2]}_{trans[3]}'


        else:
            raise NotImplementedError(f"Translation of objects with dimension {len(dim)} is not currently supported.")
        
        return im, label


    def array_rotate(self, im: np.ndarray, rot: List[int], order: int=1) -> "tuple[np.ndarray, str]":
        # TODO: Issue with low resolution not necessairly having the same dimensions
        # TODO: Look at https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.transform.Rotation.html#scipy.spatial.transform.Rotation
        # Rotation 2D
        dim = im.shape

        if len(dim) != len(rot):
            raise IndexError(f'Translation of numpy array with dimensions: {dim} is not compatible with translation {rot}')

        if len(dim) == 2:
            im = rotate(im, rot[0], order=order)
            label = f'_rot{rot[0]}'

        # Rotation 3D
        elif len(dim) == 3:
            for i in range(dim[0]):
                im[i,:,:] = rotate(im[i,:,:],rot[0], order=order)
            for i in range(dim[1]):
                im[:,i,:] = rotate(im[:,i,:],rot[1], order=order)
            for i in range(dim[2]):
                im[:,:,i] = rotate(im[:,:,i],rot[2], order=order)
            label = f'_rot{rot[0]}_{rot[1]}_{rot[2]}'

        elif len(dim) == 4:
            for i in range(dim[0]):
                im[i,:,:,:] = rotate(im[i,:,:],rot[0], order=order)
            for i in range(dim[1]):
                im[:,i,:,:] = rotate(im[:,i,:],rot[1], order=order)
            for i in range(dim[2]):
                im[:,:,i,:] = rotate(im[:,:,i],rot[2], order=order)
            for i in range(dim[3]):
                im[:,:,:,i] = rotate(im[:,:,:,i],rot[3], order=order)
            label = f'_rot{rot[0]}_{rot[1]}_{rot[2]}_{rot[3]}'
        
        else:
            raise NotImplementedError(f"Translation of objects with dimension {len(rot)} is not currently supported.")

        return im, label

    def array_scale(self, im: np.ndarray, scale: List[float], order: int=1, mode: str='symmetric', int_dims: bool=False) -> "tuple[np.ndarray, str]":
        '''Either upscales or downscales provided array

        https://scikit-image.org/docs/stable/auto_examples/transform/plot_rescale.html
        '''
        # Scaling
        dim = im.shape

        if len(dim) != len(scale):
            raise IndexError(f'Scaling of numpy array with dimensions: {dim} is not compatible with translation {scale}')

        if int_dims:
            new_dims = scale
            label = f'_si_'
        else:
            new_dims = [math.floor(x) for x in np.matmul(dim, scale)]
            label = f'_sr_'

        im = resize(im, new_dims, order=order, mode=mode) #If anti-alaising not specified, it is set to True when downsampling an image whose data type is not bool


        if len(dim) == 2:
            label = label + f'{scale[0]}_{scale[1]}'
        elif len(dim) == 3:
            label = label + f'{scale[0]}_{scale[1]}_{scale[2]}'
        elif len(dim) == 4:
            label = label + f'{scale[0]}_{scale[1]}_{scale[2]}_{scale[3]}'
        else:
            raise NotImplementedError(f"Translation of objects with dimension {len(scale)} is not currently supported.")

        return im, label
        
    def array_degrade(self, im: np.ndarray, scale: List[float], order: int=1, mode: str='symmetric', int_dims: bool=False) -> np.ndarray:
        '''Uses array_scale to scale in array down and up using the specified order'''

        dim = im.shape

        im, _ = self.array_scale(im, scale, order=order, mode=mode, int_dims=int_dims)

        im, _ = self.array_scale(im, dim, order=order, mode=mode, int_dims=True)

        return im

    
    def gen_noise(self) -> None:
        '''Add noise to provided image'''
        pass
    

    def img2patches(self, im: np.ndarray, patch: List[int], step: List[int], fname:str, min_nonzero: float = 0,
        slice_select: bool=False, slice_idx: List[int] = [], save: List[str] = [], verbose=False) -> "tuple[np.ndarray, List[str], List[int]]":
        # Depending on the number of dimenions in the `patch` value, either make 2D-4D image

        '''
        Inputs:
            save: List[str] List containing either none or two entries, with the first being the directory location to save the
                files in and the second entry containing the suffix/file type to save them as.

        Needs:
            - image: tuple of numpy arrays? That way if you want to apply the same patching to multiple different images you can?
            - min_nonzero: [float] either a flat value or a fraction of the minimum amount of input needs to be non-zero before you keep it
            - slice_select: [list of ints] a list of slices to preserve from the patches (used when pairing slices between images)
            - save_individual patches: tuple[path/filename, form] whether to save each of the patches as seperate files and what format to save them as.
                                    if the tuple is empty, then just return the stack of patches and list of filenames 
            - verbose: [bool]
            - patch; [list of ints] if they input -1 as a patch size, then take the full size of that dimension

        Returns:
            - image stack
            - the paths to the image stack or file names for each of the images in the stack
            - the slice_select list

            not_blank: List[int] a list of all the entries in the stack that passed the min_nonzero quota
        '''

        # Check patch size
        dim = im.shape

        if len(dim) != len(patch):
            raise IndexError(f'Patch selection of numpy array with dimensions: {dim} is not compatible with patch size: {patch}')

        if len(dim) != len(step):
            raise IndexError(f'Patch selection step size of numpy array with dimensions: {dim} is not compatible with step size: {step}')

        for idx, i in enumerate(patch):
            if patch[idx] > dim[idx]:
                raise ValueError(f'Patch value along dimension {[idx]} = {patch[idx]} and is larger than image along dimension {[idx]} = {dim[idx]}')

        # If they have input something for "slice_select", then they are trying to replicate patch selection
        if slice_select:
            min_nonzero = 0
            if len(slice_idx) == 0:
                print('WARNING: slice_select = True but slice_idx list is empty')

        for idx, i in enumerate(patch):
            if i == -1:
                patch[idx] = dim[idx]
                step[idx] = 1
        

        # Create a numpy stack following Pytorch protocols, so 1 dimension more than patch
        
        # Count number of non-zero entries
        cnt = 0
        blank = 0
        not_blank = []
        itter = -1

        # Get total number of patches that will be created:
        #patch_count = np.prod([len(range(0,i,step[idx])) for idx, i in enumerate(dim)])
        if verbose:
            print(f'patch guess = {np.prod([math.floor((i-patch[idx])/step[idx])+1 for idx,i in enumerate(dim)])}')
        patch_count = np.prod([math.floor((i-patch[idx])/step[idx])+1 for idx,i in enumerate(dim)])
        
        if min_nonzero > 1: #If they have given pixel/voxel numbers instead of fractions
            patch_vol = math.prod(patch) - min_nonzero
        else:
            # else, calculate the number of pixels/voxels which must be nonzero
            patch_vol = math.prod(patch) - math.prod(patch)*min_nonzero


        #TODO: There MUST be a better way to organize this whole mess, lol
        if len(dim) == 2:
            stack = np.zeros((patch_count,patch[0],patch[1]))
            if verbose:
                print(f'stack size = {stack.shape}')

            for i in range(0,dim[0],step[0]):
                for j in range(0,dim[1],step[1]):
                    if i+patch[0] <= dim[0] and j+patch[1] <= dim[1]:
                        itter = itter+1 #just a calculator for finding when blanks occur
                        samp = im[i:i+patch[0],j:j+patch[1]]

                        if min_nonzero == 0 or (samp==0).sum() <= patch_vol:
                            stack[cnt,:,:] = samp
                            cnt += 1
                            not_blank.append(itter)
                        else:
                            blank += 1
    
        elif len(dim) == 3:
            stack = np.zeros((patch_count,patch[0],patch[1], patch[2]))
            print(f'stack size = {stack.shape}')

            for i in range(0,dim[0],step[0]):
                for j in range(0,dim[1],step[1]):
                    for k in range(0,dim[2],step[2]):
                        #itter = itter+1 #just a calculator for finding when blanks occur
                        if i+patch[0] <= dim[0] and j+patch[1] <= dim[1] and k+patch[2] <= dim[2]:
                            itter = itter+1
                            samp = im[i:i+patch[0],j:j+patch[1], k:k+patch[2]]

                            if min_nonzero == 0 or (samp==0).sum() <= patch_vol:
                                stack[cnt,:,:,:] = samp
                                cnt += 1
                                not_blank.append(itter)
                            else:
                                blank += 1

        elif len(dim) == 4:
            stack = np.zeros((patch_count,patch[0],patch[1], patch[2], patch[3]))
            print(f'stack size = {stack.shape}')

            for i in range(0,dim[0],step[0]):
                for j in range(0,dim[1],step[1]):
                    for k in range(0,dim[2],step[2]):
                        for l in range(0, dim[3],step[3]):
                            #itter = itter+1 #just a calculator for finding when blanks occur
                            if i+patch[0] <= dim[0] and j+patch[1] <= dim[1] and k+patch[2] <= dim[2] and l+patch[3] <= dim[3]:
                                itter = itter+1
                                samp = im[i:i+patch[0],j:j+patch[1], k:k+patch[2], l:l+patch[2]]

                                if min_nonzero == 0 or (samp==0).sum() <= patch_vol:
                                    stack[cnt,:,:,:,:] = samp
                                    cnt += 1
                                    not_blank.append(itter)
                                else:
                                    blank += 1

        else:
            raise IndexError(f'Images of dimension {dim} not supported by this method. Only 2D-4D data accepted.')
        

        fnames = []
        pnames = [] #Pnames should be used if patching is the last process you plan on doing. These are paths
        if slice_select:
            for i in range(len(slice_idx)):
                fnames.append(f'{save[0]}{fname}_{i}')
        else:
            for i in range(cnt):
                fnames.append(f'{save[0]}{fname}_{i}')

        if verbose:
            print(f'Number of patches: {len(not_blank)}')
            print(f'Number of blank patches: {blank}')


        if save: #If they want to save the intermediate files (return a list of paths instead)
            if slice_select:
                for idx, i in enumerate(slice_idx):
                    pnames.append(self.save_image(stack[i], Path(fnames[idx]+save[1]), verbose=verbose))
            else:
                for idx, i in enumerate(fnames):
                    pnames.append(self.save_image(stack[idx], Path(fnames[idx]+save[1]), verbose=verbose))

            return np.array([]), pnames, not_blank # Send back not_blank for some comparison tests between running of this
        
        else:
            if slice_select:
                return stack[slice_idx], fnames, [] #Only return the patches which match slice_select

            else:
                return stack[:cnt], fnames, not_blank  #Only return a stack of the patches that passed the min_nonzero check
