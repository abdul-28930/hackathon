import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image
from torchvision import transforms
from ultralytics import YOLO

class gis:
    def __init__(self, image_path, model_path):
        self.gismodel = YOLO(model_path)
        self.transform = transforms.Compose([
            transforms.Resize((800, 800)),  # Resize image to 800x800
            transforms.ToTensor(),  # Convert image to tensor
        ])
        self.image = Image.open(image_path)
        self.input_tensor = self.transform(self.image).unsqueeze(0)  # Add batch dimension
    
    def predict(self):
        # Assuming gismodel is defined elsewhere
        self.pred = self.gismodel.predict(self.input_tensor)
        self.input_image = self.input_tensor.squeeze().permute(1, 2, 0).cpu().numpy()

    def pred_plotting(self, scale, a='y', b='b', c='orange', **kwargs):
        self.predict()
        # Create subplots for each class
        building_area = 0
        trees_area = 0
        extra_area = 0
        for cls in range(3):
            fig, ax = plt.subplots(**kwargs)
            # Set the title of the subplot to the class name
            ax.set_title(self.pred[0].names[cls])
            ax.imshow(self.input_image)
            ax.axis("off")
            for idx, box in enumerate(self.pred[0].boxes.xywh):
                if int(self.pred[0].boxes.cls[idx]) == cls:
                    x_center, y_center, width, height = box
                    xmin = x_center - width / 2
                    ymin = y_center - height / 2
                    xmax = x_center + width / 2
                    ymax = y_center + height / 2
                    
                    # Create a Rectangle patch
                    if cls == 0:
                        rect = patches.Rectangle((xmin, ymin), width, height, linewidth=2, edgecolor=a, facecolor=a, alpha=0.5)
                    elif cls == 1:
                        rect = patches.Rectangle((xmin, ymin), width, height, linewidth=2, edgecolor=b, facecolor=b, alpha=0.5)
                    else:
                        rect = patches.Rectangle((xmin, ymin), width, height, linewidth=2, edgecolor=c, facecolor=c, alpha=0.5)

                    pixel_to_meter = 0.01 * scale
                    area_in_meters_squared = (width * pixel_to_meter) * (height * pixel_to_meter)

                    if cls == 0:
                        building_area += area_in_meters_squared
                    elif cls == 1:
                        extra_area += area_in_meters_squared
                    else:
                        trees_area += area_in_meters_squared
                    ax.add_patch(rect)
            # Save each subplot with a unique filename
            plt.savefig(f"output images/gis_{cls}.jpg", transparent=True)
            plt.close()

        # Show the areas
        self.trees_area, self.building_area, self.extra_area = round(float(trees_area), 2), round(float(building_area), 2), round(float(extra_area), 2)
        print(f"Trees area m²: {self.trees_area}")
        print(f"Building area m²: {self.building_area}")
        print(f"Extra area m²: {self.extra_area}")

        return self.trees_area, self.building_area, self.extra_area

    def comparision_plot(self, **kwargs):
        # Areas before and after utilizing extra area
        trees_area_before = self.trees_area
        trees_area_after = self.trees_area + self.extra_area

        # Define the labels and values for the plot
        labels = ['Trees Area Before', 'Trees Area After']
        values = [trees_area_before, trees_area_after]
        plt.figure(figsize = (10,10))
        plt.bar(labels, values, color=['blue', 'green'])
        plt.ylabel('Area (m²)')
        plt.title('Area of Trees Before and After Utilizing Extra Area')
        plt.savefig("compare.jpg", transparent=True)
        plt.show()
