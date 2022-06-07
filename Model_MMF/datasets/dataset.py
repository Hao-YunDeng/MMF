#!/usr/bin/env python
# coding: utf-8

# In[1]:


import torch
import numpy as np
from mmf.common.sample import Sample
from mmf.datasets.mmf_dataset import MMFDataset


# In[2]:


class HatefulMemesFeaturesDataset(MMFDataset):
    def __init__(self, config, *args, dataset_name="hateful_memes", **kargs):
        super().__init__(dataset_name, config, *args, **kargs)       
        assert(
            self._use_features
        ), "config's 'use_features' must be true to use img dataset"        
        self.is_multilabel = self.config.get("is_multilable", False)
        
    def preprocess_sample_info(self, sample_info):
        image_path = sample_info['img'] 
        # img/1234.png -> 1234
        feature_path = image_path.split('/')[-1].split('.')[0]
        # add the feature_path key to sample_info for feature_database access
        sample_info['feature_path'] = f"{feature_path}.npy" 
        return sample_info
    
    def __getitem__(self, index):
        sample_info = self.annotation_db[index] #MMFDataset has .annotation_db field
        sample_info = self.preprocess_sample_info(sample_info)
        
        current_sample = Sample()
        
        # Haoyun: where is .text_processor??
        processed_text = self.text_processor({"text": sample_info["text"]})
        current_sample.text = processed_text["text"]
        # Haoyun: where is .update??
        if "input_ids" in processed_text:
            current_sample.update(processed_text)
            
        current_sample.id = torch.tensor(int(sample_info['id']), dtype=torch.int)    
        
        features = self.features_db.get(sample_info)
        if hasattr(self, "transformer_bbox_processor"):
            features["image_info_0"] = self.transformer_bbox_processor(
                features["image_info_0"]
            )
        current_sample.update(features)
        
        if "label" in sample_info:
            current_sample.targets = torch.tensor(sample_info["label"], dtype=torch.long)
        
        return current_sample
    
    def format_for_prediction(self, report):
        if self.is_multilabel:
            return generate_multilabel_prediction(report)
        else:
            return generate_binary_prediction(report)        


# In[3]:


class HatefulMemesImageDataset(MMFDataset):
    def __init__(self, config, *args, dataset_name="hateful_memes", **kargs):
        super().__init__(dataset_name, config, *args, **kargs)
        assert(
            self._use_features
        ), "config's 'use_features' must be true to use img dataset"        
        self.is_multilabel = self.config.get("is_multilable", False)
        
    def init_processors(self):
        super().init_processors()
        self.image_db.transfrom = self.image_processor
        
    def __getitem__(self, index):
        sample_info = self.annotation_db[index]
        current_sample = Sample()
        
        processed_text = self.text_processor({"text": sample_info["text"]})
        current_sample.text = processed_text["text"]
        if "input_ids" in processed_text:
            current_sample.update(processed_text) 
            
        current_sample.id = torch.tensor(int(sample_info['id']), dtype=torch.int)   
        
        current_sample.image = self.image_db[index]["images"][0]
        
        if "label" in sample_info:
            current_sample.targets = torch.tensor(sample_info["label"], dtype=torch.long)
        
        return current_sample   
    
    def format_for_prediction(self, report):
        if self.is_multilabel:
            return generate_multilabel_prediction(report)
        else:
            return generate_binary_prediction(report)  
        
    def visualize(self, num_samples=1, use_transforms=False, *args, **kargs):
        image_paths = []
        random_samples = np.random_randint(0, len(self), size=num_samples)
        
        for index in random_samples:
            image_paths.append(self.annotation_db[index]['img'])
            
        images = self.image_db_from_path(image_paths, use_transfomrs=use_transforms)
        visualize_images(images["images"], *args, **kargs)


# In[4]:


def generate_binary_prediction(report):
    scores = torch.nn.functional.softmax(report.scores, dim=1)
    _, labels = torch.max(scores, 1)
    probabilities = scores[:, 1]
    
    predictions = []
    for index, image_id in enumerate(report.id):
        prob = probabilities[index].items()
        label = labels[index].item()
        predictions.append({"id": image_id.item(), 
                            "prob": prob, 
                            "label": label})
    
    return predictions


# In[5]:


def generate_multilabel_prediction(report):
    scores = torch.sigmoid(report.scores)
    return [
        {"id": image_id.item(), "scores": scores[index].tolist()}
        for index, imgae_id in enumerate(report.id)
    ]


# In[6]:


#two funcs mixed up???


# In[ ]:




