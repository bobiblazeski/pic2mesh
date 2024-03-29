import argparse

from distutils.util import strtobool

def get_parser():
    parser = argparse.ArgumentParser()
    
    # General
    parser.add_argument('--data_dir', help='Tensor storage',default='./data/')

    # Renderer    
    parser.add_argument('--viewpoint_distance', type=float, default=2.25,
                        help='Distance from camera to the object')
    parser.add_argument('--viewpoint_elevation', type=float, default=0.0, 
                        help='Angle of elevation in degrees')
    parser.add_argument('--viewpoint_azimuth', type=float, default=0.0, 
                        help='No rotation so the camera is positioned on the +Z axis')
    parser.add_argument('--lights_location', nargs='+', type=float, 
                        default=[0.0, 0.0, 3.0],
                        help="Examples: -lights_location 0.0, -1.0, 3.0")

    parser.add_argument('--lights_direction', nargs='+', type=float, 
                        default=[0.0, 0.0, 3.0],
                        help="Examples: -lights_location 0.0, -1.0, 3.0")        
    parser.add_argument('--specular_color', nargs='+', type=float, 
                        default=[0.1, 0.1, 0.1])                                
                                        

    parser.add_argument('--raster_image_size', type=int, default=128,
                        help='Rasterizer image size')
    parser.add_argument('--raster_radius', type=float, default=0.01, 
                        help='Points radius')
    parser.add_argument('--raster_points_per_pixel', type=int, default=1)
    parser.add_argument('--raster_patch_size', type=int, default=512)
    parser.add_argument('--raster_max_brightness', type=float, default=0.7, 
                        help='Maximum brightness of pixel, [0-1]')
    parser.add_argument('--pulsar_gamma', type=float, default=1e-4, 
                        help='Gamma used in pulsar renderer.')

    parser.add_argument('--raster_blur_radius', type=float, default=0.0) # More than 1 black face
    parser.add_argument('--raster_faces_per_pixel', type=int, default=4)

        
    parser.add_argument('--blend_params_sigma', type=float, default=0.0001)
    parser.add_argument('--blend_params_gamma', type=float, default=0.0001)
    parser.add_argument('--blend_params_background_color', nargs='+', type=float, 
                        default=[0.0, 0.0, 0.0],
                        help="Examples: -blend_params_background_color 0.0 -1.0 3.0")

    # Discriminator
    parser.add_argument('--D_num_outcomes', help='No of discriminator outcomes', 
                        default=1)            
    parser.add_argument('--D_use_adaptive_reparam', dest='D_use_adaptive_reparam', 
                        default=True, type=lambda x: bool(strtobool(x)))
    parser.add_argument('--D_use_spectral_norm', dest='D_use_spectral_norm', 
                        default=False, type=lambda x: bool(strtobool(x)))
    parser.add_argument('--D_in_ch', type=int, help='Discriminator input channels: 3 RGB, 1 Greyscale', default=1)
    parser.add_argument('--D_out_ch',type=int, help='Generator output channels', default=256)             

    # Generator
    parser.add_argument('--G_in_ch', type=int, help='Generator input channels', default=3) # 3-points, 6+normals
    parser.add_argument('--G_out_ch',type=int, help='Generator output channels', default=256)

    parser.add_argument('--blueprint', default='blueprint_16_512.npz')
    parser.add_argument('--G_noise_amp',type=float, help='Generator noise scale.', default=0.002)
    parser.add_argument('--G_use_adaptive_reparam', dest='G_use_adaptive_reparam', 
                        default=True, type=lambda x: bool(strtobool(x)))

    parser.add_argument('--fast_generator_channels', nargs='+', type=int, 
                        default=[ 128, 128, 256, 256])
    parser.add_argument('--fast_discriminator_channels', nargs='+', type=int, 
                        default=[ 3, 256, 256])

    parser.add_argument('--initial_input_file', default='./data/mean_face_256.pth')
    parser.add_argument('--initial_input_fixed', default=True, type=lambda x: bool(strtobool(x)))

    parser.add_argument('--grid_full_size', type=int, default=32)
    parser.add_argument('--grid_slice_size', type=int, default=16)
    parser.add_argument('--style_dim', type=int, default=256)
    parser.add_argument('--synthesis_channels', nargs='+', type=int, 
                        default=[ 256, 256, 256, 256])
    parser.add_argument('--stylist_channels', nargs='+', type=int, 
                        default=[ 1, 256, 256, 256])
     
    #networks hyper parameters:
    parser.add_argument('--nfc', type=int, default=128)
    parser.add_argument('--min_nfc', type=int, default=128)
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
    parser.add_argument('--noise_amp',type=float,help='addative noise cont weight',default=0.01)
    parser.add_argument('--min_size',type=int,help='image minimal size at the coarser scale',default=25)
    parser.add_argument('--max_size', type=int,help='image minimal size at the coarser scale', default=250)

    #optimization hyper parameters:
    parser.add_argument('--niter', type=int, default=2000, help='number of epochs to train per scale')
    parser.add_argument('--gamma',type=float,help='scheduler gamma',default=0.1)    
    parser.add_argument('--lr_g', type=float, default=0.0005, help='learning rate, default=0.0005')
    parser.add_argument('--lr_d', type=float, default=0.0005, help='learning rate, default=0.0005')
    parser.add_argument('--beta1', type=float, default=0.5, help='beta1 for adam. default=0.5')
    parser.add_argument('--beta2', type=float, default=0.999, help='beta2 for adam. default=0.999')    
    parser.add_argument('--Gsteps',type=int, help='Generator inner steps',default=3)
    parser.add_argument('--Dsteps',type=int, help='Discriminator inner steps',default=3)
    parser.add_argument('--lambda_grad',type=float, help='gradient penalty weight',default=0.1)
    parser.add_argument('--alpha',type=float, help='reconstruction loss weight',default=10)

    # data module
    parser.add_argument('--data_blueprint', help='Blueprint file', default='blueprint_radial_2048.npz')
    parser.add_argument('--data_image_dir', help='Images directory', 
                        default='/home/bobi/Desktop/db/ffhq-dataset/images1024x1024/')
    parser.add_argument('--data_mask_dir', help='Image masks directory', 
                        default='/home/bobi/Desktop/face-parsing.PyTorch/res/masks')
    parser.add_argument('--data_renders_dir', help='Stl renders directory', 
                        default='/home/bobi/Desktop/pic2mesh/data/augmented/renders')

    parser.add_argument('--data_grid_dir',
                        default='/home/bobi/Desktop/pic2mesh/data/stl_grid')                        
    parser.add_argument('--data_mesh_dir',
                        default='/home/bobi/Desktop/pic2mesh/data/stl_mesh') 
    parser.add_argument('--data_image_size', help='Original image size', type=int, default=1024)
    parser.add_argument('--data_mask_size', help='Mask size', type=int, default=512)
    parser.add_argument('--data_image_resized', help='Image resized', type=int, default=256)        
    parser.add_argument('--data_style_img', help='Style image size', type=int, default=192)    
    parser.add_argument('--data_blends_no', help='How many grid files to blend', type=int, default=4)
    parser.add_argument('--data_blends_total', help='Length of the blend dataset', type=int, default=100000)

    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--pin_memory', dest='pin_memory', 
                        default=True, type=lambda x: bool(strtobool(x)))
    parser.add_argument('--shuffle', dest='shuffle', 
                        default=True, type=lambda x: bool(strtobool(x)))

    parser.add_argument('--stl_offset',type=float,help='Mesh scaling offset',default=0.02)
    
    parser.add_argument('--fast_outline_size', type=int, default=32)
    parser.add_argument('--fast_baseline_size', type=int, default=128)
    parser.add_argument('--fast_image_root',
        default='/home/bobi/Desktop/db/ffhq-dataset/images1024x1024')
    parser.add_argument('--fast_render_root',
        default='/home/bobi/Desktop/db/renders')        
    parser.add_argument('--mask_root',
        default='/home/bobi/Desktop/db/ffhq-dataset/masks')
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--fast_image_size', type=int, default=128)
    parser.add_argument('--mask_size', type=int, default=512)    
    parser.add_argument('--fast_image_mean', type=float, default=0.5)
    parser.add_argument('--fast_image_std', type=float, default=0.5)
    parser.add_argument('--geoaug_policy', default='scaling,translation') 
    parser.add_argument('--diffaug_policy', default='color,translation')
    
    # Logger
    parser.add_argument('--log_grid_samples', help='Log image grid number of samples', type=int, default=8)
    parser.add_argument('--log_grid_rows', help='Log image grid number of rows', type=int, default=4)
    parser.add_argument('--log_grid_padding', help='Log image grid number padding', type=int, default=2)
    parser.add_argument('--log_scale_each', dest='Scale each image from the grid', 
                            default=False, type=lambda x: bool(strtobool(x)))
    parser.add_argument('--log_pad_value', help='Value for the padded pixels', type=int, default=0)
    parser.add_argument('--log_batch_interval', help='Image logging interval', type=int, default=250)
    parser.add_argument('--log_mesh_interval', help='Mesh export interval', type=int, default=20)
    parser.add_argument('--log_render_interval', help='Image logging interval', type=int, default=100)
    
    # Losses    
    parser.add_argument('--mesh_edge_loss_weight', type=float, default=1.00, help='Mesh edge loss')    
    parser.add_argument('--mesh_normal_consistency_weight', type=float, default=0.01, help='Mesh edge loss')
    parser.add_argument('--mesh_laplacian_smoothing_weight', type=float, default=0.01, help='Mesh edge loss')
    parser.add_argument('--mesh_laplacian_smoothing_method', default='uniform', 
                        help='Laplacian smoothing type uniform, cot or cotcurv')
    return parser