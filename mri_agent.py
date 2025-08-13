import os
import sys
import SimpleITK as sitk
import numpy as np
import pandas as pd
import torch
from transformers import ViTImageProcessor 
from train_sequence import ViTWithFC 
from PIL import Image
from basic_agent import BasicAgent
from mydataloader import label_encoder as encoder
import shutil
from pathlib import Path

os.environ["TRANSFORMERS_VERBOSITY"] = "error"
os.environ["nnUNet_raw"] = "/tmp/nnUNet_raw"
os.environ["nnUNet_preprocessed"] = "/tmp/nnUNet_preprocessed"
os.environ["nnUNet_results"] = "/tmp/nnUNet_results"

series_organ_dict={'sag_t1_mprage': ['brain', 'thyroid']}

foundation_path = os.path.join(os.path.dirname(__file__),"foundation")  # Path to foundation models

def run_foundation(mri_path, series, task):
    original_dir = os.getcwd() 
    filename = os.path.basename(mri_path)
    subj = filename.split('.')[0]
    ext='.nii.gz'
    destination_path = foundation_path+'/nifti_data_dir/'+subj+'/'+subj+'_' +series+ext 
    os.makedirs(os.path.dirname(destination_path), exist_ok=True)
    shutil.copy2(mri_path, destination_path)  # preserves metadata
    if foundation_path not in sys.path:
        sys.path.insert(0, foundation_path)
    os.chdir(foundation_path+'/fm_mediumweight/ml/')
    output_dir = foundation_path+'/fm_mediumweight/ml/' + 'output_masks'
    from fm_mediumweight.ml.pipeline import run_segmentation_pipeline
    run_segmentation_pipeline(subj, series, task, dicom_dir="dicom_data_dir", nifti_dir=foundation_path +"/nifti_data_dir", output_dir=output_dir)
    os.chdir(original_dir)
    return output_dir+'/brain/'+subj+'_'+series+'.csv'


class MRIReaderAgent(BasicAgent):
    def perceive(self, image_path):
        """Ingest and possibly interpret input. which series"""
        series = self.__infer_series(image_path) 
        print(f"it looks like this is {series} MR series")
        return series[0]

    def planning(self, series):
        # plan which tasks to run in foundation model
        task='organs'
        if series == "sag_t1_mprage":
            task='brain'
        elif(series == "ax_t2_blade_ctm_chest_abdomen_pelvis"):
            task='organs'
        elif(series=="ax_t1_oop"):
            task='vertebrae'
        return task

    def register_tool(self, name, function):
        #run foundation model
        self.tools[name] = function
        print(f"[{self.name}] Tool registered: {name}")

    def feedback(self):
        #use internet to check if the values are good
        #if values are too small that suggests that there is segmentation error. clean up.
        print("checking the volumes... ")

    def run(self, mri_path: str) -> str:
        print(f"[{self.name}] Reading MRI from {mri_path}...")
        # Placeholder for MRI analysis
        series = self.perceive(mri_path)
        output_dir = self.tools['segment_and_volume'](mri_path, series, self.planning(series))
        findings = "Detected moderate brain atrophy and white matter lesions."
        findings, prompt = self.__summary_report(output_dir)
        return findings, prompt

    def __summary_report(self, output_dir):
        def generate_prompt():    
            prompt = f"""
        You are provided with subject-level measurements for brain regions compared to population statistics.

        Your task:
        - For each region, state whether the subject is "above", "below", or "within normal range" (±1 SD)
        - Highlight any potential trends across the regions
        - Write a short narrative summary

        Please respond in clear, structured language.
        """
            return prompt
        
        prompt = generate_prompt()
        merged_df = self.__add_population(output_dir)
        findings = merged_df.to_markdown(index=True)
        """
        response = client.chat.completions.create(
            model="gpt-4",
            messages=[{"role": "user", "content": prompt}]
        )
        findings = response.choices[0].message.content
        """
        return findings, prompt

    def __add_population(self, output_dir):
        df = pd.read_csv("models/population.csv")
        mean = df.mean(numeric_only=True).rename(lambda x: f"{x}")
        std = df.std(numeric_only=True).rename(lambda x: f"{x}")
        df1 = mean.to_frame(name='population_mean')
        df2 = std.to_frame(name='population_std')
        merged = df1.join(df2)
        subject = pd.read_csv(output_dir)
        #----
        subject['Label'] = subject['Label'].apply(lambda x: x[0] if isinstance(x, list) else x)
        merged.index = merged.index.map(lambda x: x[0] if isinstance(x, list) else x)
        subject = subject.set_index('Label')
        merged['subject_value'] = subject['Volume (mm³)']
        #----
        merged.columns = ['population_mean', 'population_std', 'subject_value']
        merged.index = subject.index
        return merged
    def __infer_series(self, dicom_path):
        def preprocess_data(dicom_path):
            image_processor = ViTImageProcessor.from_pretrained("google/vit-base-patch16-224")
            im = sitk.ReadImage(dicom_path)
            image = sitk.GetArrayFromImage(im)
            num_slices = image.shape[0]
            random_indices = np.random.choice(num_slices, size=10, replace=True)
            image = image[random_indices]
            pil_images = [Image.fromarray((image[i]).astype("uint8")).convert("RGB") for i in range(10)]
            image_list=[image_processor(images=image, return_tensors="pt")["pixel_values"]  for image in pil_images]
            return image_list

        model_path = os.path.join(os.path.dirname(__file__),"models","vit_with_fc.pth")  # Path to saved model weights
        model = ViTWithFC(num_classes=35)
        checkpoint = torch.load(model_path)
        model.load_state_dict(checkpoint)
        model.eval()  # Set to evaluation mode

        image_list = preprocess_data(dicom_path)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        images = image_list[0].to(device)
        model.to(device)
        with torch.no_grad():
            outputs = model(images)
            predictions = torch.argmax(outputs, dim=1)  # Get predicted classes
        
        series = encoder.inverse_transform(predictions.cpu().numpy())
        return series

if __name__ == "__main__":
    agent=MRIReaderAgent()
    agent.register_tool('segment_and_volume',run_foundation)
    mri_output = agent.run('data/Dataset/imagesTs/ec18dc53-faca-4d56-80d6-c538eaaac234_sag_t1_mprage_0000.nii.gz')
    context_memory.add("MRIReader", "Findings", mri_output)
    answer = qa_agent.run("Is there evidence of neurodegeneration?", context_memory.get_context())
