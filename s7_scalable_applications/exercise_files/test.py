import glob 
import os
imgs_path=os.getcwd()
print("hello")
print(imgs_path)
name_list = glob.glob("data\\*\\*.jpg")

print(len(name_list))