from huggingface_hub import hf_hub_download
# from transformers import AutoImageProcessor, TableTransformerForObjectDetection
from transformers import DetrImageProcessor, DetrForObjectDetection
import torch
from PIL import Image

# you can specify the revision tag if you don't want the timm dependency
processor = DetrImageProcessor.from_pretrained('facebook/detr-resnet-50', revision='no_timm')
model = DetrForObjectDetection.from_pretrained('facebook/detr-resnet-50', revision='no_timm')

# # file_path = hf_hub_download(repo_id='nielsr/example-pdf', repo_type='dataset', filename='example_pdf.png')
scenes = [('00:21:48', '00:23:40'), ('00:23:40','00:23:55'), ('00:23:56', '00:25:59'), ('00:26:00','00:27:07')]
frame_list = [(169, 1049), (2745,), (5417, 5815), (7239,)] # clothes, glasses, hemlet

res = [{'content_id': 1260036017, 
        'fragments': []}]
for scene, frames in zip(scenes, frame_list):
    print(scene, frames)
    for id in frames:
        file_path = '/Users/huguanqing/Desktop/hack24/1260036017_frames/{:08d}.png'.format(id)

        image = Image.open(file_path).convert('RGB')
        inputs = processor(images=image, return_tensors='pt')
        outputs = model(**inputs)

        # convert outputs (bounding boxes and class logits) to COCO API
        # let's only keep detections with score > 0.9
        target_sizes = torch.tensor([image.size[::-1]])
        results = processor.post_process_object_detection(outputs, target_sizes=target_sizes, threshold=0.95)[0]

        for i, score, label, box in zip(range(len(results)), results['scores'], results['labels'], results['boxes']):
            box = [round(i, 2) for i in box.tolist()]
            print(
                    f'Detected {model.config.id2label[label.item()]} with confidence '
                    f'{round(score.item(), 3)} at location {box}'
            )
            x1,y1,x2,y2 = [int(x) for x in box]
            save_path = f'{id:08d}_{i}.png'
            image.crop((x1, y1, x2, y2)).save(save_path) 
            res[0]['fragments'].append({
                'start_time': scene[0],
                'end_time': scene[1],
                'picture_link': save_path
            })

open('res.txt', 'w').write(str(res))

        
