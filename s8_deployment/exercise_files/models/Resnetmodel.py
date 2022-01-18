import torch 
import torchvision.models as models
from torchvision import transforms
from PIL import Image
import numpy as np

model = models.resnet18(pretrained=True)
script_model = torch.jit.script(model)
script_model.save('deployable_model.pt')

## makes randome noise input picture for the resnet model
def get_white_noise_image(w,h):
    pil_map = Image.fromarray(np.random.randint(0,255,(w,h,3),dtype=np.dtype('uint8')))
    return pil_map

rand_img = get_white_noise_image(229,299)

#making sure the picture is as expected
#rand_img.show()

# pre procseeing of the input picture,
preprocess = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )])

#these models needs to be put into a evaluation state for prediction and evaluation.
model.eval = False
script_model.eval = False 


#processing the image for evaluation
rand_img_preprocessed = preprocess(rand_img)

## making extra dimension for batch size. 
batch_img_cat_tensor = torch.unsqueeze(rand_img_preprocessed, 0)


model_output = model(batch_img_cat_tensor)
script_model_output = script_model(batch_img_cat_tensor)

## comparing whole output tensor. 
assert torch.allclose(model_output,script_model_output),f"Model output does not corrospond."