import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from shapely.geometry import Polygon, box, MultiPolygon
import ast
import geopandas as gpd
from skimage import measure
from sam_processing import SamProcessing

class RefShapeProcessing(SamProcessing):
    
    def __init__(self, image_path, bbox):
        super().__init__(image_path, bbox)
        pass
    
    def load_clean_paddle_pickle(self):
        self.df_paddle = pd.read_pickle(self.paddle_pickle)
        self.df_paddle['Text_score'] = self.df_paddle['Text_score'].round(2)
        
        # Convert boxes and Poly Columns in DF_2 into python list
        self.df_paddle['boxes'] = self.df_paddle['boxes'].apply(lambda x: ast.literal_eval(x) if isinstance(x, str) else x)
        self.df_paddle['Poly'] = self.df_paddle['Poly'].apply(lambda x: ast.literal_eval(x) if isinstance(x, str) else x)
        
    def load_sam_pickle(self):
        self.df_sam = self.best_result_df
        
    def is_percent_inside(self, row1, row2, threshold=0.30):
        """
        Check if Poly from df_2 is sufficiently inside segment from df_1.
        Also checks bounding box overlap.
        """
        # Segment polygon from df_1
        seg_poly = Polygon(zip(row1['segments']['x'], row1['segments']['y']))

        # Reference polygon and box from df_2
        ref_poly = Polygon(row2['Poly'])
        ref_box_coords = row2['boxes'] 
        ref_box = box(*ref_box_coords)

        # Sanity check for valid geometry
        if not seg_poly.is_valid or not ref_poly.is_valid:
            return False

        # Check polygon overlap ratio
        inter_area = seg_poly.intersection(ref_poly).area
        ref_area = ref_poly.area
        if ref_area == 0 or (inter_area / ref_area) < threshold:
            return False

        # Optional: check if reference box overlaps with segment polygon's bounding box
        seg_bounds = seg_poly.bounds  # (minx, miny, maxx, maxy)
        seg_box = box(*seg_bounds)
        
        if not seg_box.intersects(ref_box):
            return False

        return True
    
    def match_ref_to_shape(self) -> None:
        # Initialize columns for matched references and their Text_scores
        self.df_sam['matched_references'] = [[] for _ in range(len(self.df_sam))]
        self.df_sam['matched_text_scores'] = [[] for _ in range(len(self.df_sam))]
        self.df_sam['reference'] = None  # first matched reference
        self.df_sam['first_text_score'] = None  # first matched text score

        for i1, row1 in self.df_sam.iterrows():
            matches = []
            scores = []
            for i2, row2 in self.df_paddle.iterrows():
                if self.is_percent_inside(row1, row2):
                    matches.append(row2['reference'])
                    scores.append(row2['Text_score'])

            self.df_sam.at[i1, 'matched_references'] = matches
            self.df_sam.at[i1, 'matched_text_scores'] = scores
            self.df_sam.at[i1, 'reference'] = matches[0] if matches else None
            self.df_sam.at[i1, 'first_text_score'] = scores[0] if scores else None

        
        self.df_sam.iat[3, 5] = self.df_sam.iat[4, 5]
        self.df_sam.iat[3, 6] = self.df_sam.iat[4, 6]
        self.df_sam.iat[3, 7] = self.df_sam.iat[4, 7]
        self.df_sam.iat[3, 8] = self.df_sam.iat[4, 8]
        
        self.df_toshape = 'df_to_shape.pkl'
        
        self.df_sam.to_pickle(self.df_toshape)
        
    
    def geo_pandas(self):
        
        df = self.df_sam
        # Convert 'segments' to shapely Polygons
        df['geometry'] = df['segments'].apply(lambda seg: Polygon(zip(seg['x'], seg['y'])))

        # Drop 'segments' to avoid column size issues
        gdf = gpd.GeoDataFrame(df.drop(columns=['segments']), geometry='geometry')

        # Set CRS 
        #gdf.set_crs(epsg=3857, inplace=True)  

        # Save as GeoPackage
        gdf.to_file("Geopandas/segments.gpkg", layer='segmented_polygons', driver="GPKG")
        
        # Load the GeoPackage file
        gdf = gpd.read_file("Geopandas/segments.gpkg", layer="segmented_polygons")

        display(gdf.head(5))
        
        # Plot full GeoDataFrame with color by 'reference'
        ax = gdf.plot(column="reference", cmap="Set2", edgecolor="black", figsize=(10, 8))

        # Overlay row 2 in red
        #gdf.loc[[2]].plot(ax=ax, edgecolor='red', facecolor='none', linewidth=2)


        #centroid = gdf.loc[2, 'geometry'].centroid

        plt.savefig("Geopandas/Geopandas_map.png", dpi=300, bbox_inches='tight')

        # Show the plot
        plt.show()
        
    def process(self):
        print("\n--- Running Geopandas Pipeline ---")

        # 1. Run all inherited processing (Paddle → SAM)
        super().process()

        # 2. Load cleaned PaddleOCR output
        print("\n[4] Loading cleaned PaddleOCR pickle...")
        self.load_clean_paddle_pickle()

        # 3. Load best SAM results
        print("\n[5] Loading best SAM results...")
        self.load_sam_pickle()

        # 4. Match Paddle references to SAM segments
        print("\n[6] Matching references to shapes...")
        self.match_ref_to_shape()

        # 5. Create GeoDataFrame, export to GPKG and plot
        print("\n[7] Creating and saving GeoDataFrame...")
        self.geo_pandas()

        print("\n Geopandas Pipeline completed successfully.")


