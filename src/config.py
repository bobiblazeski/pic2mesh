import argparse

from distutils.util import strtobool

def get_parser():
    parser = argparse.ArgumentParser()
    
    # General
    parser.add_argument('--data_dir', help='Tensor storage',default='./data/')

    # Renderer    
    parser.add_argument('--viewpoint_distance', type=float, default=2.0, 
                        help='Distance from camera to the object')
    parser.add_argument('--viewpoint_elevation', type=float, default=0.0, 
                        help='Angle of elevation in degrees')
    parser.add_argument('--viewpoint_azimuth', type=float, default=0.0, 
                        help='No rotation so the camera is positioned on the +Z axis')
    parser.add_argument('--lights_location', nargs='+', type=float, 
                        default=[0.0, -1.0, 3.0],
                        help="Examples: -lights_location 0.0, -1.0, 3.0")

    parser.add_argument('--raster_image_size', type=int, default=128, 
                        help='Rasterizer image size')
    parser.add_argument('--raster_radius', type=float, default=0.006, 
                        help='Points radius')
    parser.add_argument('--raster_points_per_pixel', type=int, default=4)
    parser.add_argument('--raster_patch_size', type=int, default=256)
    parser.add_argument('--raster_max_brightness', type=float, default=0.7, 
                        help='Maximum brightness of pixel, [0-1]')

    # Discriminator
    parser.add_argument('--D_num_outcomes', help='No of discriminator outcomes', 
                        default=32)    
    parser.add_argument('--D_filters', nargs='+', type=int, 
                        default=[3, 32, 64],
                        help="Examples: -D_filters 3 32 64")
    parser.add_argument('--D_act_name', 
                        help='Discriminator activation name Swish or LeakyReLU',
                        default='Swish')
    parser.add_argument('--D_use_adaptive_reparam', dest='D_use_adaptive_reparam', 
                        default=True, type=lambda x: bool(strtobool(x)))
    parser.add_argument('--D_use_spectral_norm', dest='D_use_spectral_norm', 
                        default=False, type=lambda x: bool(strtobool(x)))
    
    # Stylist
    parser.add_argument('--backbone', default='vgg11')
    
    # Generator
    parser.add_argument('--dlatent_size', type=int, default=128)
    parser.add_argument('--in_channel', type=int, default=6) # 3 just points, 6 normals too
    parser.add_argument('--out_channel',type=int,help='kernel size',default=32)

    parser.add_argument('--blueprint', default='blueprint127.npz')
    parser.add_argument('--G_noise_amp',type=float, help='Generator noise scale.', default=0.003)
    parser.add_argument('--G_use_adaptive_reparam', dest='G_use_adaptive_reparam', 
                        default=True, type=lambda x: bool(strtobool(x)))
    
    #networks hyper parameters:
    parser.add_argument('--nfc', type=int, default=32)
    parser.add_argument('--min_nfc', type=int, default=32)
    parser.add_argument('--ker_size',type=int,help='kernel size',default=3)
    parser.add_argument('--num_layer',type=int,help='number of layers',default=5)
    parser.add_argument('--stride',help='stride',default=1)
    parser.add_argument('--padd_size',type=int,help='net pad size',default=1)#math.floor(opt.ker_size/2)
    
    #parser.add_argument('--mode', help='task to be done', default='train')
    #workspace:
    parser.add_argument('--not_cuda', action='store_true', help='disables cuda', default=0)
    
    #load, input, save configurations:
    parser.add_argument('--netG', default='', help="path to netG (to continue training)")
    parser.add_argument('--netD', default='', help="path to netD (to continue training)")
    parser.add_argument('--manualSeed', type=int, help='manual seed')
    parser.add_argument('--nc_z',type=int,help='noise # channels',default=3)
    parser.add_argument('--nc_im',type=int,help='image # channels',default=3)
    parser.add_argument('--out',help='output folder',default='Output')
        
    #pyramid parameters:
    parser.add_argument('--scale_factor',type=float,help='pyramid scale factor',default=0.75)#pow(0.5,1/6))
    parser.add_argument('--noise_amp',type=float,help='addative noise cont weight',default=0.1)
    parser.add_argument('--min_size',type=int,help='image minimal size at the coarser scale',default=25)
    parser.add_argument('--max_size', type=int,help='image minimal size at the coarser scale', default=250)

    #optimization hyper parameters:
    parser.add_argument('--niter', type=int, default=2000, help='number of epochs to train per scale')
    parser.add_argument('--gamma',type=float,help='scheduler gamma',default=0.1)
    parser.add_argument('--lr_g', type=float, default=0.0005, help='learning rate, default=0.0005')
    parser.add_argument('--lr_d', type=float, default=0.0005, help='learning rate, default=0.0005')
    parser.add_argument('--beta1', type=float, default=0.5, help='beta1 for adam. default=0.5')
    parser.add_argument('--Gsteps',type=int, help='Generator inner steps',default=3)
    parser.add_argument('--Dsteps',type=int, help='Discriminator inner steps',default=3)
    parser.add_argument('--lambda_grad',type=float, help='gradient penelty weight',default=0.1)
    parser.add_argument('--alpha',type=float, help='reconstruction loss weight',default=10)

    # data module
    parser.add_argument('--data_blueprint', help='Blueprint file', default='./data/blueprint127.npz')
    parser.add_argument('--data_image_dir', help='Images directory', 
                        default='/home/bobi/Desktop/db/ffhq-dataset/images1024x1024')
    parser.add_argument('--data_mask_dir', help='Image masks directory', 
                        default='/home/bobi/Desktop/face-parsing.PyTorch/res/masks')
    parser.add_argument('--data_image_size', help='Original image size', type=int, default=1024)
    parser.add_argument('--data_mask_size', help='Mask size', type=int, default=512)
    parser.add_argument('--data_image_resized', help='Image resized', type=int, default=256)
    parser.add_argument('--data_patch_size', help='Patch size', type=int, default=128)
    parser.add_argument('--data_style_img', help='Style image size', type=int, default=192)

    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--num_workers', type=int, default=4)
    

    return parser