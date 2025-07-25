import ultralytics
import torch
from ultralytics import SAM
import matplotlib.pyplot as plt
from ultralytics.models.sam import SAM2Predictor
from ultralytics.models.sam import Predictor
from ultralytics.models.sam import Predictor as SAMPredictor
from PIL import Image
import cv2
import pandas as pd
import numpy as np
import os
from image_processing import ImageProcessing
from paddle_processing import PaddleProcessing


torch.cuda.empty_cache()
torch.cuda.get_device_name(0)

class SamProcessing(PaddleProcessing):
    
    names = ['Sam2_Base_Output', 'Sam2_Large_Output', 'Mobile_Sam_Output']
    modls = ['sam2_b.pt', 'sam2_l.pt', 'mobile_sam.pt']
    
    def __init__(self, image_path, bbox):
        super().__init__(image_path, bbox)
        self.sam_result_file = []
        self.dfs = []
        
        
    def sam_results(self, name, model_path):
        
        print(f'STARTING : {name}')


        model = SAM(model_path)
        results = model.predict(source = self.skeleton_img, bboxes = self.bbox)
        
        for result in results:
            result.save_crop(save_dir=f"{name}", file_name="detection")
            img_with_masks = result.plot()
            img_rgb = img_with_masks[..., ::-1]

            plt.imshow(img_rgb)
            plt.axis('off')
            plt.show()

            self.sam_df = result.to_df()
            print(f'DISPLAY the Dataframe for {name}')
            display(self.sam_df.head())

            img = cv2.imread(self.trim_image)  
            img_with_segments = img.copy()

            for idx, row in self.sam_df.iterrows():
                seg = row['segments']
                polygon = np.array(list(zip(seg['x'], seg['y']))).astype(np.int32)
                cv2.polylines(img_with_segments, [polygon], isClosed=True, color=(0, 255, 0), thickness=2)

            img_rgb = cv2.cvtColor(img_with_segments, cv2.COLOR_BGR2RGB)
            plt.imshow(img_rgb)
            plt.axis('off')
            plt.title("All Segments")
            plt.show()
            
            self.res = f"{name}/results.pkl"
            self.sam_result_file.append(self.res) 

            self.sam_df.to_pickle(self.res)
            print(f"DataFrame saved to {name}/results.pkl")
            
            
    def get_results(self) -> None:
        
        for name, modl in zip(self.names, self.modls):
            self.sam_results(name = name, model_path = modl)
            
            
    def get_highest_confidence(self) -> None:
        self.dfs =[]
        print(len(self.sam_result_file))
        
        for i in self.sam_result_file:
            df = pd.read_pickle(i)
            self.dfs.append(df)
        
        self.samb_df, self.saml_df, self.msam_df = self.dfs[0], self.dfs[1], self.dfs[2]
        
        confidences = pd.DataFrame({'samb_df': self.samb_df['confidence'], 
                                    'saml_df': self.saml_df['confidence'],
                                    'msam_df': self.msam_df['confidence']})
        
        # Find which DataFrame has the max confidence for each row
        max_conf_df = confidences.idxmax(axis=1)  

        # Collect rows from each DataFrame where it has the highest confidence
        filtered_rows = []

        for df_name, df in zip(['samb_df', 'saml_df', 'msam_df'], [self.samb_df, self.saml_df, self.msam_df]):
            mask = max_conf_df == df_name
            filtered_rows.append(df[mask])
         
        # Combine all selected rows back together
        self.best_result_df = pd.concat(filtered_rows).sort_index().reset_index(drop=True)
        
        self.best_result_df.to_pickle("best_SAM_results.pkl")
        print("DataFrame saved to best_SAM_results.pkl")
     
    def plot_mask(self) -> None:

        # Load the original image
        self.img = cv2.imread(self.trim_image)  
        img_with_segments = self.img.copy()
        
        for df in [self.samb_df, self.saml_df, self.msam_df]:
        

            for idx, row in df.iterrows():
                seg = row['segments']

                # Convert x/y lists to a Nx2 array of (x, y) points
                polygon = np.array(list(zip(seg['x'], seg['y']))).astype(np.int32)

                # Draw polygon on image
                cv2.polylines(img_with_segments, [polygon], isClosed=True, color=(0, 255, 0), thickness=2)


            # Convert BGR to RGB for matplotlib display
            img_rgb = cv2.cvtColor(img_with_segments, cv2.COLOR_BGR2RGB)

            # Show the result
            plt.imshow(img_rgb)
            plt.axis('off')
            plt.title("All Segments")
            plt.show()
            
    def save_mask(self) -> None:
        for name, df in zip(self.names, self.dfs):
            # Folder to save the extracted segments
            output_folder = name
            os.makedirs(output_folder, exist_ok=True)
            
            for idx, row in df.iterrows():
                seg = row['segments']
                polygon = np.array(list(zip(seg['x'], seg['y']))).astype(np.int32)

                img = self.img
                # Create a mask for the polygon
                mask = np.zeros(img.shape[:2], dtype=np.uint8)
                cv2.fillPoly(mask, [polygon], 255)

                # Extract bounding rect of polygon
                x, y, w, h = cv2.boundingRect(polygon)

                # Crop image and mask to bounding rectangle
                cropped_img = img[y:y+h, x:x+w]
                cropped_mask = mask[y:y+h, x:x+w]

                # Apply mask to the cropped image (keep only polygon area)
                extracted = cv2.bitwise_and(cropped_img, cropped_img, mask=cropped_mask)

                # Optional: To get a transparent background, create BGRA image with alpha channel
                bgra = cv2.cvtColor(extracted, cv2.COLOR_BGR2BGRA)
                bgra[:, :, 3] = cropped_mask  # Set alpha channel from mask

                # Save the extracted object as PNG with transparency
                save_path = os.path.join(output_folder, f"segment_{idx}.png")
                cv2.imwrite(save_path, bgra)

                print(f"Saved segment {idx} to {save_path}")

                # Plot the segment
                plt.figure()
                plt.imshow(cv2.cvtColor(bgra, cv2.COLOR_BGRA2RGBA))
                plt.title(f"Segment {idx}")
                plt.axis('off')
                plt.show()

    def process(self):
        print("\n--- Running SAM Segmentation Pipeline ---")

        super().process()


        print("\n[1] Running all SAM models...")
        self.get_results()

        print("\n[2] Selecting highest confidence segments...")
        self.get_highest_confidence()

        print("\n[3] Plotting all segmentation masks...")
        self.plot_mask()

        print("\n[4] Saving segmented mask outputs...")
        self.save_mask()

        print("\n SAM pipeline completed successfully.")



