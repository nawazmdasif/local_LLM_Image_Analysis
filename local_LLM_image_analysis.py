
import os
import glob
import base64
from io import BytesIO
from PIL import Image
import requests
import json
from filelock import FileLock
import pandas as pd
import time
import spacy
import re
from datetime import datetime
import cv2
import numpy as np
from typing import Optional, Tuple, Dict, List

try:
    nlp = spacy.load('en_core_web_sm')
except OSError:
    print("Please install spaCy model using: python -m spacy download en_core_web_sm")
    exit()

class ShopAnalyzer:
    def __init__(self, api_base: str, model_name: str, max_retries: int = 3):
        self.api_base = api_base
        self.model_name = model_name
        self.max_retries = max_retries
        self.session = requests.Session()
        self.feature_extractor = cv2.SIFT_create()

    def extract_features_from_text(self, text: str) -> Dict[str, List[str]]:
        """Extract features from text using NLP techniques."""
        doc = nlp(text.lower())
        features = {
            'products': [],
            'attributes': [],
            'shop_types': [],
            'banners': []
        }
        
        for ent in doc.ents:
            if ent.label_ in ['PRODUCT', 'ORG']:
                features['products'].append(ent.text)
        
        return features

    def analyze_shop_image(self, image_data: str, image_path: str) -> Optional[str]:
        headers = {'Content-Type': 'application/json'}
        prompt = """
        Please provide a detailed analysis of this shop image. Include:
        1. Shop name if visible
        2. Exact shop type(s) seen (pharmacy, grocery, clothing, etc.)
        3. List ALL visible products, especially FMCG items
        4. ALL visible payment/mobile banking/telecom banners
        5. Describe fridges/refrigerators if present
        6. Describe shutters/security features if present
        7. Note shop age indicators
        8. Note shop size indicators
        Mention everything you see in detail, as this will be used for accurate classification.
        """
        
        data = {
            "model": self.model_name,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {"type": "image_url", "image_url": f"data:image/jpeg;base64,{image_data}"}
                    ]
                }
            ]
        }

        for attempt in range(self.max_retries):
            try:
                response = self.session.post(f"{self.api_base}/chat/completions", json=data, headers=headers)
                response.raise_for_status()
                content = response.json()
                if 'choices' in content and len(content['choices']) > 0:
                    return content['choices'][0]['message']['content']
                return None
            except requests.exceptions.RequestException as e:
                print(f"Attempt {attempt + 1} failed: {e}")
                if attempt < self.max_retries - 1:
                    time.sleep(2 ** attempt)
                continue
        return None

    def parse_response(self, response: str) -> Dict:
        try:
            text_blocks = response.lower().split('\n')
            criteria = {
                "shop_name": "unknown",
                "pharmacy": "n", "grocery": "n", "super_shop": "n", 
                "electronics": "n", "telecom": "n", "general_shop": "n", 
                "cosmetics": "n", "clothing": "n",
                "has_fmcg": "n", "has_fridge": "n", "has_shutter": "n",
                "shop_age": "unknown", "shop_size": "unknown",
                "banner_bkash": "n", "banner_nagad": "n", "banner_robi": "n",
                "banner_gp": "n", "banner_others": "n"
            }

            # Keywords for accurate matching
            keywords = {
                'pharmacy': ['medicine', 'pharmacy', 'medical store', 'drug', 'pharmaceutical'],
                'grocery': ['grocery', 'food items', 'vegetables', 'fruits'],
                'super_shop': ['supermarket', 'super shop', 'department store'],
                'electronics': ['electronics', 'gadgets', 'mobile phones'],
                'telecom': ['telecom', 'mobile operator', 'sim card'],
                'clothing': ['clothing', 'garments', 'fashion', 'apparel'],
                'cosmetics': ['cosmetics', 'beauty products', 'makeup'],
                'fmcg': ['packaged food', 'beverages', 'toiletries', 'household items',
                        'snacks', 'soft drinks', 'personal care', 'dairy'],
                'fridge': ['refrigerator', 'fridge', 'freezer', 'cooler'],
                'shutter': ['shutter', 'rolling shutter', 'metal door', 'security gate'],
                'banners': {
                    'bkash': ['bkash', 'b-kash'],
                    'nagad': ['nagad'],
                    'robi': ['robi'],
                    'gp': ['grameenphone', 'gp']
                }
            }

            # Process each text block
            for block in text_blocks:
                # Check shop types
                for shop_type in ['pharmacy', 'grocery', 'super_shop', 'electronics', 
                                'telecom', 'clothing', 'cosmetics']:
                    if any(kw in block for kw in keywords[shop_type]):
                        criteria[shop_type] = 'y'

                # Check FMCG
                if any(kw in block for kw in keywords['fmcg']):
                    criteria['has_fmcg'] = 'y'

                # Check fridge
                if any(kw in block for kw in keywords['fridge']):
                    criteria['has_fridge'] = 'y'

                # Check shutter
                if any(kw in block for kw in keywords['shutter']):
                    criteria['has_shutter'] = 'y'

                # Check banners
                for banner, kws in keywords['banners'].items():
                    if any(kw in block for kw in kws):
                        criteria[f'banner_{banner}'] = 'y'

            # Extract shop name using NLP
            doc = nlp(response)
            potential_names = []
            for ent in doc.ents:
                if ent.label_ == "ORG":
                    potential_names.append(ent.text)
            
            if potential_names:
                criteria["shop_name"] = potential_names[0]

            # Process size and age
            size_indicators = {
                'big': ['large', 'spacious', 'huge', 'multiple floors'],
                'medium': ['medium', 'moderate', 'average'],
                'small': ['small', 'compact', 'tiny']
            }

            age_indicators = {
                'new': ['new', 'modern', 'renovated', 'fresh'],
                'old': ['old', 'traditional', 'worn out', 'aging']
            }

            for block in text_blocks:
                for size, indicators in size_indicators.items():
                    if any(indicator in block for indicator in indicators):
                        criteria['shop_size'] = size
                        break
                
                for age, indicators in age_indicators.items():
                    if any(indicator in block for indicator in indicators):
                        criteria['shop_age'] = age
                        break

            return criteria
        except Exception as e:
            print(f"Error in parse_response: {e}")
            return criteria

    def process_single_image(self, image_path: str, output_path: str, lock: FileLock) -> bool:
        try:
            print(f"Processing image: {image_path}")
            
            # Convert image to base64
            with Image.open(image_path) as img:
                buffered = BytesIO()
                img.convert('RGB').save(buffered, format='JPEG')
                image_data = base64.b64encode(buffered.getvalue()).decode()

            # Get analysis from model
            response = self.analyze_shop_image(image_data, image_path)
            if not response:
                return False

            # Parse the response
            criteria = self.parse_response(response)
            
            # Extract image info
            folder_name = os.path.basename(os.path.dirname(image_path))
            filename = os.path.basename(image_path)
            match = re.match(r'image_(\d+)_(\d{4}-\d{2}-\d{2})\..*', filename)
            shop_id = match.group(1) if match else "unknown"
            date = match.group(2) if match else "unknown"

            # Save analysis
            with lock:
                self.save_analysis(output_path, image_path, shop_id, date, folder_name, criteria, response)
                
            return True
        except Exception as e:
            print(f"Error processing {image_path}: {e}")
            return False

    def save_analysis(self, output_path: str, image_path: str, shop_id: str, 
                     date: str, folder_name: str, criteria: Dict, analysis: str):
        try:
            df_new = pd.DataFrame([{
                "Image Path": image_path,
                "Folder": folder_name,
                "Shop ID": shop_id,
                "Date": date,
                "Shop Name": criteria["shop_name"],
                "Pharmacy": criteria["pharmacy"],
                "Grocery": criteria["grocery"],
                "Super-shop": criteria["super_shop"],
                "Electronics": criteria["electronics"],
                "Telecom": criteria["telecom"],
                "General Shop": criteria["general_shop"],
                "Cosmetics": criteria["cosmetics"],
                "Clothing": criteria["clothing"],
                "FMCG": criteria["has_fmcg"],
                "Fridge": criteria["has_fridge"],
                "Shutter": criteria["has_shutter"],
                "bKash Banner": criteria["banner_bkash"],
                "Nagad Banner": criteria["banner_nagad"],
                "Robi Banner": criteria["banner_robi"],
                "GP Banner": criteria["banner_gp"],
                "Shop Age": criteria["shop_age"],
                "Shop Size": criteria["shop_size"],
                "Analysis": analysis,
                "Timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
            }])

            if os.path.exists(output_path):
                df_existing = pd.read_excel(output_path)
                df_combined = pd.concat([df_existing, df_new], ignore_index=True)
            else:
                df_combined = df_new

            df_combined = df_combined.sort_values(['Folder', 'Date'])
            df_combined.to_excel(output_path, index=False)

        except Exception as e:
            print(f"Error saving analysis: {e}")

    def process_images(self, images_dir: str, output_path: str) -> Tuple[int, int]:
        if not os.path.exists(images_dir):
            print(f"Images directory not found: {images_dir}")
            return 0, 0

        image_patterns = [
            os.path.join(images_dir, '**/*.[jJ][pP][gG]'),
            os.path.join(images_dir, '**/*.[jJ][pP][eE][gG]'),
            os.path.join(images_dir, '**/*.[pP][nN][gG]')
        ]

        image_paths = []
        for pattern in image_patterns:
            image_paths.extend(glob.glob(pattern, recursive=True))

        image_paths.sort(key=lambda x: (os.path.basename(os.path.dirname(x)), x))

        total = len(image_paths)
        if total == 0:
            print(f"No images found in {images_dir}")
            return 0, 0

        print(f"Found {total} images to process")
        lock = FileLock(f"{output_path}.lock")
        successful = 0

        for i, path in enumerate(image_paths, 1):
            print(f"Processing image {i}/{total} from folder: {os.path.basename(os.path.dirname(path))}")
            if self.process_single_image(path, output_path, lock):
                successful += 1

        return successful, total

def main():
    config = {
        "api_base": "http://127.0.0.1:11434/v1",
        "model_name": "llama3.2-vision:latest",
        "images_dir": "images",
        "output_path": "20250213_Shop_Image_Analysis_02'feb-to 03 mar.xlsx"
    }

    analyzer = ShopAnalyzer(config["api_base"], config["model_name"])
    successful, total = analyzer.process_images(config["images_dir"], config["output_path"])
    print(f"Analysis complete. Processed {successful}/{total} images successfully")

if __name__ == "__main__":
    main()
