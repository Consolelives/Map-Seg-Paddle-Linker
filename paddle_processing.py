from paddleocr import PaddleOCR
from matplotlib import pyplot as plt 
import cv2 
import os 
import paddleocr
from pprint import pprint
import re
import pandas as pd
import numpy as np
from PIL import Image
import json
import os
import shutil
from image_processing import ImageProcessing

    

class PaddleProcessing(ImageProcessing):
    folder_path = "paddle_out"
    
    def __init__(self, image_path, bbox):
        super().__init__(image_path, bbox) 
        self.json_file = None
        self.data = None
        self.filtered = None
        self.df =  None

    
    def instantiate_paddle(self):
        # Delete the folder before saving new information
        if os.path.exists(self.folder_path) and os.path.isdir(self.folder_path):
            shutil.rmtree(self.folder_path)
        
        ocr = PaddleOCR(lang = 'en', 
                        use_doc_orientation_classify=False, 
                        use_doc_unwarping=False,
                        use_textline_orientation=False)
        
        # Run OCR inference on a image 
        result = ocr.predict(input=self.trim_image)
        fol_path = "paddle_out"
        # Visualize the results and save the JSON results
        for res in result:
            
            #res.print()
            res.save_to_img(fol_path)
            res.save_to_json(fol_path)
        
    def load_json(self):
        for file in os.listdir(self.folder_path):
            if file.endswith(".json"):
                self.json_file = os.path.join(self.folder_path, file)
                break
                
        # Open and load JSON file
        with open(self.json_file, "r", encoding="utf-8") as f:
            self.data = json.load(f)
            
    def fix_key(self, k): 
        """
        Corrects OCR errors where '/' is misread as '1' or missing.

        The OCR model often confuses '/' with '1' or omits it completely.
        This function processes the relevant list inside the dictionary,
        replacing every third item equal to '1' with '/',
        or inserting '/' at the third position if neither '1' nor '/' is present.

        Args:
            data_dict (dict): Dictionary containing OCR results, 
                            including the list to clean.

        Returns:
            dict: Updated dictionary with corrected OCR output.
        """
        
        third_char = k[2]
        
        if third_char == '/':
            return k
        elif third_char == '1':
            # Replace the 3rd char (index 2) with '/'
            return k[:2] + '/' + k[3:]
        else:
            # Insert '/' at position 2
            return k[:2] + '/' + k[2:]
    
    def clean_json_data(self):
        rec_texts, rec_boxes, rec_scores, rec_polys = (
            self.data.get(key, []) for key in ("rec_texts", "rec_boxes", "rec_scores", "rec_polys"))
        
        combined_2 = {}
        
        for texts, boxes, polys, scores in zip(rec_texts, rec_boxes, rec_polys, rec_scores):
            combined_2[texts] = {'boxes':boxes, 'Poly': polys, 'Text_Scores': scores}
        
        # Remove ref numbers lower than six characters
        filtered_combined_2 = {k: v for k, v in combined_2.items() if len(k) >= 6}
        
        # Remove ref numbers ending with 'ha'
        pattern = re.compile(r"^\d")
        filtered_combined_2 = {k: v for k, v in filtered_combined_2.items() if pattern.match(k) and not k.endswith('ha')}
        
        # This uses the function outside the class
        self.filtered = {self.fix_key(k): v for k, v in filtered_combined_2.items()}
    
    def save_res_pandas(self):
        result = []
        
        for key, val in self.filtered.items():
            result.append({
                'reference': key,
                "Poly": val["Poly"],
                "Text_score": val["Text_Scores"],
                "boxes": val["boxes"]
            })
        
        self.df_paddle = pd.DataFrame(result)
        
        # Create a new column
        self.df_paddle['Coordinates_str'] = self.df_paddle['boxes'].apply(lambda x: '-'.join(map(str, x)))
        
        self.paddle_pickle = f"{self.folder_path}/paddle_ocr_dataframe.pkl"
        
        # Save to pickle file
        self.df_paddle.to_pickle(self.paddle_pickle)
        
    def process(self):
        print("\n--- Running PADDLE PADDLE Pipeline ---")
        super().process()
        self.instantiate_paddle()
        self.load_json()
        self.clean_json_data()
        self.save_res_pandas()
        
        print("\n PADDLE PADDLE pipeline completed successfully.")

        