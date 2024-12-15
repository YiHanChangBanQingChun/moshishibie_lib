import os
from PIL import Image, ImageOps

def find_and_merge_images(base_path, keyword, output_path):
    # 初始化一个空列表来存储找到的图片
    images = []

    # 遍历1到12的文件夹
    for i in range(1, 13):
        folder_path = os.path.join(base_path, f'features_{i}')
        if os.path.exists(folder_path):
            # 遍历文件夹中的所有文件
            for file_name in os.listdir(folder_path):
                if keyword in file_name and file_name.endswith('.png'):
                    # 打开图片并添加到列表中
                    img_path = os.path.join(folder_path, file_name)
                    img = Image.open(img_path)
                    # 添加白色边框
                    img_with_border = ImageOps.expand(img, border=3, fill='white')
                    images.append(img_with_border)

    # 检查是否找到了12张图片
    if len(images) != 12:
        print(f"找到的图片数量不足12张，仅找到{len(images)}张。")
        return

    # 获取单张图片的宽度和高度
    img_width, img_height = images[0].size

    # 创建一个新的空白图片，大小为4张图片的宽度和3张图片的高度
    merged_image = Image.new('RGB', (img_width * 4, img_height * 3))

    # 将图片拼接到新的空白图片上
    for index, img in enumerate(images):
        x = (index % 4) * img_width
        y = (index // 4) * img_height
        merged_image.paste(img, (x, y))

    # 保存拼接后的图片
    merged_image.save(output_path)
    print(f"拼接后的图片已保存至 {output_path}")

# 使用示例
base_path = r'd:\Users\admin\Documents\MATLAB\moshishibie_lib\第四次作业-高光谱降维与分类\outputs\feature_variation'
keyword = '深度学习'  # 替换为你要查找的关键词
output_path = r'd:\Users\admin\Documents\MATLAB\moshishibie_lib\第四次作业-高光谱降维与分类\outputs\feature_variation\深度学习.png'
find_and_merge_images(base_path, keyword, output_path)