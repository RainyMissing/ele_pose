from ultralytics import YOLO
from PIL import Image
import matplotlib.pyplot as plt



# Load a model

model = YOLO('./test/test.pt')  # load a custom model

# Predict with the model
results = model('./test/test.jpg')  # predict on an image


img_path = './test/test.jpg'
image = Image.open(img_path)
plt.figure(figsize=(10, 10))
print(type(results))

skeleton = [
            [10, 9],
            [10, 8],
            [9, 7],
            [8, 7],
            [8, 6],
            [7, 5],
            [6, 4],
            [5, 2],
            [4, 3],
            [4, 2],
            [2, 1],
            [3, 1],
        ]

for i, r in enumerate(results):
    # Plot results image
    im_bgr = r.plot()  # BGR-order numpy array
    im_rgb = Image.fromarray(im_bgr[..., ::-1])  # RGB-order PIL image


    data_tensor = r.keypoints.xyn  


    data_numpy = data_tensor.cpu().numpy()

    plt.imshow(image)

    for data in data_numpy:
        # 绘制关键点
        for k in data[:4]:
            plt.scatter(k[0] * image.width, k[1] * image.height, color='red')  # 调整坐标以匹配图片尺寸
        for k in data[4:8]:
            plt.scatter(k[0] * image.width, k[1] * image.height, color='green')  # 调整坐标以匹配图片尺寸
        
        for k in data[8:12]:
            plt.scatter(k[0] * image.width, k[1] * image.height, color='blue')  # 调整坐标以匹配图片尺寸
        
        # 绘制骨架
        for link in skeleton:
            start_index, end_index = link[0] - 1, link[1] - 1  # 索引调整为从0开始
            plt.plot(
                [data[start_index][0] * image.width, data[end_index][0] * image.width],
                [data[start_index][1] * image.height, data[end_index][1] * image.height],
                'green'
            )

    plt.axis('off')  # 关闭坐标轴
    

    # 保存图片到磁盘
    save_path = './test/result_test.jpg'  # 更改为你想要保存的路径和文件名
    plt.savefig(save_path, bbox_inches='tight', pad_inches=0.0)
    plt.close() 