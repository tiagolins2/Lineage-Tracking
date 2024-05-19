import sys
#import pyperclip
import imageio
from torchvision import models
import os

#sys.path.append("/content/sample_data")
#from frond_tracking79 import*
from frond_tracking88_test import*
from get_paths_by_id import*
#from lineage_tree import*

from ultralytics import YOLO
model = SiameseNetwork()

## load frond matching model
model.load_state_dict(torch.load("siamese_model3.pth",map_location=torch.device('cpu')))
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#filename1= "/content/drive/MyDrive/images/model-testing/JD2_6wp_cropped/image_camA_0_1_20220829-220116_6.png";
#filename2= "/content/drive/MyDrive/images/model-testing/JD2_6wp_cropped/image_camA_0_1_20220830-070117_6.png";
#filename1 = "/content/sample_data/separate_images/image_camA_0_0_20221019-194008_Plate2_a4.png";
#filename2 = "/content/sample_data/separate_images/image_camA_0_0_20221020-194008_Plate2_a4.png"

## load frond finding model
model_fd = YOLO("last.pt"); # a slightly improved model

#path="/content/sample_data/separate_images"; #path='/content/drive/MyDrive/images/model-testing/JD2_6wp_cropped';

C_params= [1.12050277, 1.15364989, -5.09743406, -2.52339094, 0.54034491, 4.10348772, -1.88356986, 1.87468465, -1.87611715, -1.0326488, 4.02585197, 3.53109731,
           0.28623279, 0.62377714, -0.06587951, 1.01947809, 2.7334802, 1.80356796, -2.26209321, 2.91891391, -2.58517694, -1.68343237, -0.46842763, 0.53871813, -1.67982886]

def write_matrix_indices(matrix, output_file, num):
    with open(output_file, 'w') as file:
        #file.write("row_index\tcolumn_index\n")
        for i, row in enumerate(matrix):
            for j, value in enumerate(row):
                if value == num:
                    file.write(f"{i}\t{j}\n")


def run_tracking_model(path, model_fd, model, device, timestep_value, timestep_unit, radial_threshold, C_params=C_params, C_iou=2, C_d=0.1, C_iou_d=1, Ci_iou=20, C_siam=10, C_f=1):

  
  addresses_by_id = read_addresses_by_id(os.path.join(path, 'output.csv'))
   
  ## read file containing all address and wellIDs:
  for id_, addresses_info in addresses_by_id.items():
        print(f"Processing images by ID: {id_}")
        list_files = [];
        for address_info in addresses_info:
          address, cam_info, plate_info, last_chars = address_info
          list_files.append(address);
        #print(f"Address: {address}, Cam Info: {cam_info}, Plate Info: {plate_info}, Last Chars: {last_chars}")

 
        #list_files = addresses;
        #list_files = [ x for x in list_files if wellID1 in x ]
        #list_files = [ x for x in list_files if wellID2 in x ]
        #radial_threshold=200 for 12, 400 for 6
        list_files.sort();
        #print(list_files)
        global_id = np.ones([100,300+2])*-1 #30 frames
        global_id_output = np.empty((101, 100), dtype='object')#np.ones([101,100+2])*-1
  
        figures = []
        saving_address = path+"/frond_tracking/"+cam_info+"_"+plate_info+"_"+last_chars[:-4]; 
        print(list_files);
        no_mask_detection = 0; 
        for i,filename in enumerate(list_files):
          
          if os.path.exists(saving_address): # where files will be saved
            pass;
          else:
            os.mkdir(saving_address)
          if i==0:
            filename1= filename
            image1, masks1, centroids_set1 = get_image_info_expand(filename1, model_fd, radial_threshold); img1=image1.copy()
            mini_masks = create_mini_masks(masks1.copy(),image1.copy()); #save_mini_masks(mini_masks.copy(), filename)
            global_id[0:len(masks1),0] = range(len(masks1))
            timing_id = np.zeros([len(masks1),1]);
            global_id[0:len(masks1),2+i*3] = range(len(masks1)) #2,5,8,
            global_id_output[1:len(masks1)+1,0] = range(len(masks1))
            global_id_output[0,0] = f"0 {timestep_unit}"
            color_map = abs(np.random.rand(len(masks1),3));
            for j in range(len(masks1)):
              global_id[j,3+i*3] = centroids_set1[j][0] #6,9,12
              global_id[j,4+i*3] = centroids_set1[j][1] #7,10,13
            print(filename)
            img2plot = plot_color_fronds(img1, masks1, color_map, global_id[0:len(masks1),2+i*3]);
            for jj in range(len(centroids_set1)):
              idx = np.where(global_id[:,2+(i)*3]==jj);
              plt.text(centroids_set1[jj][0]-(415-radial_threshold),centroids_set1[jj][1]-(415-radial_threshold),f"{idx[0][0]}",horizontalalignment='center')
              print("number of masks for 1st image=")
              print(len(masks1))  
            if len(masks1)>0:
              pass;
            else:
              no_mask_detection=1;
              i = 0;
  
            plt.imshow(extract_center(img2plot, radial_threshold*2, radial_threshold*2));
            plt.axis("off")
            plt.savefig(saving_address+"/"+f'plot_{i}.png')
            plt.show()
            plt.clf() 
            #plt.imshow(image1); plt.show()
          else:
            filename2 = filename
            print(filename) 
            # pass the masks and centroids to the main_function instead and then reorder them based on their global id
            image2, masks2, centroids_set2 = get_image_info_expand(filename2, model_fd, radial_threshold); img2=image2.copy()
            if len(masks2)>0 and no_mask_detection==0:  # if empty, skip
             mini_masks = create_mini_masks(masks2.copy(),image2.copy()); #save_mini_masks(mini_masks.copy(), filename);

             matrix, matrix2, siam_dist, ass_siam, original_iou, iou_matrix,distances,siamese_distance, forces = main_function(image1, masks1, centroids_set1, image2, masks2, centroids_set2, model_fd,model, device,C_params, C_iou, C_d, C_iou_d, Ci_iou, C_siam, C_f)
             print(f"size of matrix {matrix2.shape}")
             area_set1 = np.array([np.sum(mask == 1) for mask in masks1]); area_set1_sum=np.sum(area_set1); area_set1=area_set1/area_set1_sum
             area_set2 = np.array([np.sum(mask == 1) for mask in masks2]); area_set2_sum=np.sum(area_set2); area_set2=area_set2/area_set2_sum
             set1 = [[ii*kk, jj*kk] for [ii, jj], kk in zip(centroids_set1,area_set1)];
             set2 = [[ii*kk, jj*kk] for [ii, jj], kk in zip(centroids_set2,area_set2)];
             #print(f"set1={set1}")
             #print(f"set2={set2}")
             set1 = [sum(pair)/len(pair) for pair in zip(*set1)]
             set2 = [sum(pair)/len(pair) for pair in zip(*set2)]
             #print(f"set1={set1}")
             #print(f"set2={set2}")

             filename_save = cam_info+ plate_info+ last_chars[0:4]
           # np.savetxt(f'/content/sample_data2/{filename_save}/mat.txt', (matrix2).astype(int), delimiter='  ', fmt='%d');
           # np.savetxt(f'/content/sample_data2/{filename_save}/original_iou.txt', (original_iou), delimiter='  ');
           # np.savetxt(f'/content/sample_data2/{filename_save}/iou_matrix.txt', (iou_matrix), delimiter='  ');
           # np.savetxt(f'/content/sample_data2/{filename_save}/distances.txt', (distances), delimiter='  ');
           # np.savetxt(f'/content/sample_data2/{filename_save}/siamese_distance.txt', (siamese_distance), delimiter='  ');
           # np.savetxt(f'/content/sample_data2/{filename_save}/forces.txt', (forces), delimiter='  ');
           # np.savetxt(f'/content/sample_data2/{filename_save}/avg_displacement.txt', [set1, set2], delimiter='  ');
           # np.savetxt(f'/content/sample_data2/{filename_save}/avg_growth.txt', [area_set1_sum, area_set2_sum], delimiter='  ');
           # matrix_indices_file = f'/content/sample_data2/{filename_save}/matrix_indices.txt';
           # write_matrix_indices(matrix2, matrix_indices_file, 1)
           # budding_indices_file = f'/content/sample_data2/{filename_save}/budding_indices.txt';
           # write_matrix_indices(matrix2, budding_indices_file, 2)

             for pair in ass_siam: ## assignment to global_id
               x = np.where(global_id[:,2+(i-1)*3] == pair[0])
               global_id[x,2+(i)*3] = pair[1];
               global_id_output[x[0]+1,i] = x[0]
               global_id_output[0,i] = f"{timestep_value*i} {timestep_unit}s";

               for j in range(len(pair)):
                 global_id[x,3+i*3] = centroids_set2[pair[1]][0] #6,9,12 #pair[1] contains the index of set2 that matches with set1
                 global_id[x,4+i*3] = centroids_set2[pair[1]][1] #7,10,13
                  
 
            ## add new ids for new daughter fronds:
             set1, set2 = np.where(matrix2==2);
             if len(set1)>0:
               for mset1, mset2 in zip(set1, set2):
                 new_element = int(np.max(global_id[:,0]))+1;
                 global_id[new_element,0] = new_element
                 timing_id=np.append(timing_id, i);
                 global_id[new_element,2+(i)*3] = mset2;
                 global_id_output[new_element+1,i] = new_element; 
                 #find global id of parent frond:
                 x_mother=np.where(global_id[:,2+(i-1)*3] == mset1)
                 global_id[new_element,1] = x_mother[0][0];
                 append_array = color_map[x_mother[0][0],:]+np.random.uniform(-0.2, 0.2, 3)
                 color_map = abs(np.vstack([color_map, append_array]));
                 color_map[color_map>1] = 1
             #print(filename)

            ##plotting
             img2plot = plot_color_fronds(img2, masks2, color_map, global_id[:,2+i*3])
             plt.imshow(extract_center(img2plot, radial_threshold*2, radial_threshold*2));
             for jj in range(len(centroids_set2)):
               idx = np.where(global_id[:,2+(i)*3]==jj);
               if idx[0].size>0:
                 plt.text(centroids_set2[jj][0]-(415-radial_threshold),centroids_set2[jj][1]-(415-radial_threshold),f"{idx[0][0]}",horizontalalignment='center') #the original with global ids
               #plt.text(centroids_set2[jj][0]-(415-h/2),centroids_set2[jj][1]-(415-h/2),f"{jj}",horizontalalignment='center')

             plt.axis("off")
             plt.savefig(saving_address+"/"+f'plot_{i}.png')
             plt.show()
             plt.clf() 
            #print(global_id)

            #import shutil
            #origin = f'/content/sample_data2/{os.path.basename(filename1)[:-4]}'
            #target = f'/content/sample_data2/{os.path.basename(filename2)[:-4]}'
            # Fetching the list of all the files
            #files = os.listdir(origin)

            # Fetching all the files to directory
            #for file_name in files:
              #if '.png' in file_name and os.path.basename(filename1)[:-4] in file_name:
               # shutil.copy(origin+'/'+file_name, target+'/'+file_name)
            #shutil.copy(f'/content/plot_{i-1}.png', target+'/plot_0.png')
            #shutil.copy(f'/content/plot_{i}.png', target+'/plot_1.png')
            #shutil.make_archive(f'/content/sample_data2/{(filename)[:-4]}', 'zip', f'/content/sample_data2/{(filename)[:-4]}')

             filename1 = filename2; masks_old=masks1.copy();
             image1= image2.copy(); masks1= masks2.copy(); centroids_set1=centroids_set2.copy()
            else:
             print("skipped one image");
             no_mask_detection=1;
        global_id_output[global_id_output == None] = ''
        np.savetxt(f'{saving_address}/global_id.csv', (global_id_output), delimiter=',', fmt='%s');
        data = global_id[0:int(np.max(global_id[:,0]))+1,0:2];
        #create_tree(color_map, data, timing_id, saving_address); 

  # Create the GIF
  #images = [];
  #for j in range(i+1):
  #  images.append(imageio.imread(f"/content/plot_{j}.png"))
  #imageio.mimsave('/content/plot.gif',images,fps=3)
  #print(original_iou)
  #print(iou_matrix)
  #print(distances)
  #print(siamese_distance)
  return global_id, timing_id, color_map
### generate the frame-to-frame matrix relating the masks
C_param = [1.12050277, 1.15364989, -5.09743406, -2.52339094, 0.54034491, 4.10348772, -1.88356986, 1.87468465, -1.87611715, -1.0326488, 4.02585197, 3.53109731,
           0.28623279, 0.62377714, -0.06587951, 1.01947809, 2.7334802, 1.80356796, -2.26209321, 2.91891391, -2.58517694, -1.68343237, -0.46842763, 0.53871813, -1.67982886]
C_param = [12.77560247,4.93068095,-5.28510945,-6.23532926, 5.8240042, -3.86117708, -6.98716583, 9.56647709, -0.20309498,-3.60374143,-1.38448283,3.34975351,4.5494971,-12.11002078,1.51883858,-19.77901457,10.23598679,2.44110108,-6.66826775,-12.45349468,-4.03156457, -11.36879312, -12.09276248,-2.77700921,-3.21107141];

path = input("Enter the full path of the directory containing addresses.txt: ")

radial_threshold = input("Enter radial threshold: (100 for 96-wp, 400 for 6-wp) "); radial_threshold=int(radial_threshold);
timestep_value = input("Enter timestep value: "); timestep_value=int(timestep_value);
timestep_unit = input("Enter timestep unit: ");
if os.path.exists(path+"/frond_tracking"):
  pass;
else:
  os.mkdir(path+"/frond_tracking")
 

global_id, timing_id, color_map = run_tracking_model(path, model_fd, model, device, timestep_value, timestep_unit, radial_threshold, C_param, C_iou=2, C_d=0.5, C_iou_d=2, Ci_iou=10, C_siam=10, C_f=1)




