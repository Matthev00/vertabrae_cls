from monai.transforms import Compose, LoadImaged, EnsureChannelFirstd, Spacingd, Orientationd, NormalizeIntensityd, EnsureTyped, ScaleIntensityd
import torch
from config import SEG_MODEL_DIR
import SimpleITK as sitk

from monai.bundle import ConfigParser
import tempfile
import os
from monai.inferers import sliding_window_inference



class VertebraExtractor:
    """
    A class to extract specific vertebrae from CT images.
    """
    def __init__(self, is_full_resolution: bool = False, device: torch.device = torch.device("cuda")):
        """
        Initialize the VertebraExtractor.

        Args:
            is_full_resolution (bool): If True, use full resolution model.
            device (torch.device): The device to run the model on.

        """
        pixdim = (1.5, 1.5, 1.5) if is_full_resolution else (3.0, 3.0, 3.0)
        self.transforms = Compose([
            LoadImaged(keys=["image"]),
            EnsureTyped(keys=["image"]),
            EnsureChannelFirstd(keys=["image"]),
            Orientationd(keys=["image"], axcodes="RAS"),
            Spacingd(keys=["image"], pixdim=pixdim, mode="bilinear"),
            NormalizeIntensityd(keys=["image"], nonzero=True),
            ScaleIntensityd(keys=["image"], minv=-1.0, maxv=1.0)
        ])
        self.device = device
        self.is_full_resolution = is_full_resolution
        self.model = self._load_model(is_full_resolution)

    
    def _load_model(self) -> torch.nn.Module:
        """
        Load the vertebra extraction model.

        Returns:
            torch.nn.Module: The loaded model.
        """

        model_filename = "model.pt" if self.is_full_resolution else "model_lowres.pt"
        model_path = SEG_MODEL_DIR / "models" / model_filename
        config_path = SEG_MODEL_DIR / "config" / "inference.json"

        parser = ConfigParser()
        parser.read_config(config_path)

        model = parser.get_parsed_content("network_def")
        weights = torch.load(model_path, map_location=self.device)

        if isinstance(weights, dict) and "state_dict" in weights:
            weights = weights["state_dict"]
        model.load_state_dict(weights)
        model.eval()
        return model
    
    def _save_tensor_as_nrrd(self, tensor: torch.Tensor, metadata: dict, filename:str) -> None:
        """
        Save a PyTorch tensor as an NRRD file with metadata.

        Args:
            tensor (torch.Tensor): The tensor to save.
            metadata (dict): Metadata to include in the NRRD file.\
            filename (str): The filename to save the NRRD file as.
        """
        array = tensor.numpy()  
        
        image = sitk.GetImageFromArray(array)
        
        if "spacing" in metadata:
            image.SetSpacing(metadata["spacing"])
        if "direction" in metadata:
            image.SetDirection(metadata["direction"])
        if "origin" in metadata:
            image.SetOrigin(metadata["origin"])
        
        sitk.WriteImage(image, filename)

    def _transform_data(self, tensor: torch.Tensor, metadata: dict) -> torch.Tensor:
        """
        Transform the input tensor data.

        Args:
            tensor (torch.Tensor): The input tensor.
            metadata (dict): Metadata for the tensor.

        Returns:
            torch.Tensor: The transformed tensor.
        """
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_file = os.path.join(temp_dir, "output.nrrd")
            
            self._save_tensor_as_nrrd(tensor, metadata, temp_file)
            
            data = {"image": temp_file}
            
            transformed_data = self.transforms(data)
            return transformed_data["image"]
    
    def get_segmentation(self, raw_tensor: torch.Tensor, metadata: dict) -> torch.Tensor:
        """
        Get the segmentation of the input tensor.

        Args:
            raw_tensor (torch.Tensor): The input tensor.
            metadata (dict): Metadata for the tensor.

        Returns:
            torch.Tensor: The segmentation result.
        """
        transformed_tensor = self._transform_data(raw_tensor, metadata)
        
        with torch.inference_mode():
            input_tensor = transformed_tensor.unsqueeze(0).to(self.device)
            output = sliding_window_inference(
                inputs=input_tensor,
                roi_size=(96, 96, 96),  
                sw_batch_size=1,
                predictor=self.model,
                padding_mode="replicate",
                overlap=0.5,
                mode="gaussian"
            )
            output = self.model(transformed_tensor)
            output = torch.argmax(output, dim=1).squeeze(0).cpu().numpy()
        
        return output

