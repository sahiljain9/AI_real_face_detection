import streamlit as st
import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image

st.set_page_config(
    page_title="AI Generated Image Detection",
    page_icon="🤖",
    layout="centered"
)

class ImprovedCNN(nn.Module):

    def __init__(self):
        super().__init__()

        self.features = nn.Sequential(

            nn.Conv2d(3,32,3,padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(32,64,3,padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(64,128,3,padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(128,256,3,padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),

            nn.AdaptiveAvgPool2d((1,1))

        )

        self.classifier = nn.Sequential(

            nn.Flatten(),

            nn.Linear(256,128),

            nn.ReLU(),

            nn.Dropout(0.5),

            nn.Linear(128,1)

        )

    def forward(self,x):

        x=self.features(x)

        x=self.classifier(x)

        return x


device=torch.device("cpu")

model=ImprovedCNN()

model.load_state_dict(
    torch.load(
        "best_cnn_model.pth",
        map_location=device
    )
)

model.eval()

transform=transforms.Compose([

    transforms.Resize((224,224)),

    transforms.ToTensor(),

    transforms.Normalize(
        [0.485,0.456,0.406],
        [0.229,0.224,0.225]
    )

])

st.title("🤖 AI Generated Image Detection")

st.write("Upload an image to classify whether it is AI Generated or Real.")

uploaded_file=st.file_uploader(
    "Upload Image",
    type=["jpg","jpeg","png"]
)

if uploaded_file is not None:

    image=Image.open(uploaded_file).convert("RGB")

    st.image(image,width=350)

    img=transform(image).unsqueeze(0)

    with torch.no_grad():

        output=model(img)

        probability=torch.sigmoid(output).item()

    if probability>=0.5:

        st.success("Prediction : AI Generated Image")

    else:

        st.success("Prediction : Real Image")

    st.write(f"Confidence : {probability:.4f}")
