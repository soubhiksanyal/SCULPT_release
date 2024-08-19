# Author: Soubhik Sanyal
# License: Same as the SCULPT_release github repo
# These are some utility functions written to 
# label fashion images used in SCULPT using CLIP and BLIP.
# One needs to install CLIP and BLIP models in addtion to
# the other requirements to use these functions.


import os
import numpy as np
import matplotlib.pyplot as plt
import torch
from PIL import Image
from tqdm import tqdm
import clip
import json
import ipdb

def clip_labelling_of_zalando():
    ## This gives almost 97% accuracy for seleccitng the t-shirt or shirt clothing type
    img_folder = '/is/cluster/fast/ssanyal/project_4/data/zalando/clo_img_seg_original_excluding_jacket_blazer'#'/is/cluster/work/ssanyal/project_4/data/stylegan3/zalando/RGB_with_alpha/imgs/clo_imgs_seg' # '/is/cluster/fast/ssanyal/project_4/data/zalando/512x512_with_alpha_samples' # 
    img_paths = os.listdir(img_folder) # os.listdir(img_folder)[:100] #

    output_lables_json = '/is/cluster/fast/ssanyal/project_4/data/zalando/clip_labelling/clothing_labels_excluding_jacket_blazer_withoitaug_full_16362_CLIP.json'

    # img_paths = ['RB022G002-K11@18.png']

    # cloth_class_text = 'person wearing short shirt and short pants'

    # template_texts = [
    #     'a bad photo of a {}',
    #     # 'a good photo of a {}',
    #     # 'a low resolution photo of a {}',
    #     # 'a high resolution photo of a {}',
    #     'a no background photo of a {}',
    #     'a fashion photo of a {}',
    #     'a photo of a {}',
    #     'a blurry photo of a {}',
    #     'a photo of a cool {}',
    #     'a photo of a single {}',
    #     'a segmented photo of a {}',
    # ]

    # cloth_class_text_list = [
    #     'a person wearing long t-shirt and short pants', 
    #     'a person wearing short t-shirt and short pants', 
    #     'a person wearing long t-shirt and long pants', 
    #     'a person wearing short t-shirt and long pants', 
    #     'a person wearing long shirt and long pants', 
    #     'a person wearing long shirt and short pants', 
    #     'a person wearing short shirt and long pants', 
    #     'a person wearing short shirt and short pants', 
    #     'a person wearing long open blazer and long pants', 
    #     'a person wearing long open blazer and short pants',
    #     'a person wearing long close blazer and long pants',
    #     'a person wearing long close blazer and short pants', 
    #     'a person wearing long open jacket and long pants',
    #     'a person wearing long open jacket and short pants',
    #     'a person wearing long close jacket and long pants',
    #     'a person wearing long close jacket and short pants',
    #     ]

    # cloth_class_text_list = [
    #     'a person wearing long sleev t-shirt and short pants', 
    #     'a person wearing short sleev t-shirt and short pants', 
    #     'a person wearing long sleev t-shirt and long pants', 
    #     'a person wearing short sleev t-shirt and long pants', 
    #     'a person wearing long sleev shirt and long pants', 
    #     'a person wearing long sleev shirt and short pants', 
    #     'a person wearing short sleev shirt and long pants', 
    #     'a person wearing short sleev shirt and short pants', 
    #     'a person wearing long sleev open blazer and long pants', 
    #     'a person wearing long sleev open blazer and short pants',
    #     'a person wearing long sleev close blazer and long pants',
    #     'a person wearing long sleev close blazer and short pants', 
    #     'a person wearing long sleev  open jacket and long pants',
    #     'a person wearing long sleev open jacket and short pants',
    #     'a person wearing long sleev close jacket and long pants',
    #     'a person wearing long sleev close jacket and short pants',
    #     ]

    # cloth_class_text_list_upper = [
    #     'a person is wearing long sleev t-shirt', 
    #     'a person is wearing short sleev t-shirt', 
    #     'a person is wearing long sleev t-shirt', 
    #     'a person is wearing short sleev t-shirt', 
    #     'a person is wearing long sleev shirt', 
    #     'a person is wearing long sleev shirt', 
    #     'a person is wearing short sleev shirt', 
    #     'a person is wearing short sleev shirt', 
    #     'a person is wearing long sleev open blazer', 
    #     'a person is wearing long sleev open blazer',
    #     'a person is wearing long sleev close blazer',
    #     'a person is wearing long sleev close blazer', 
    #     'a person is wearing long sleev  open jacket',
    #     'a person is wearing long sleev open jacket',
    #     'a person is wearing long sleev close jacket',
    #     'a person is wearing long sleev close jacket',
    #     ]

    cloth_class_text_list_upper = [
        'the person is wearing a t-shirt', 
        'the person is wearing a shirt', 
        # 'the person is wearing a blazer',
        # 'the person is wearing a jacket',
        ]

    cloth_class_text_list_upper_blazer_jacket = [
        'the upper body clothing is open', 
        'the upper body clothing is close', 
        ]

    cloth_class_text_list_upper_sleev = [
        'long sleeves', 
        'short sleeves', 
        ]

    cloth_class_text_list_lower = [
        'the person is wearing long pants', 
        'the person is wearing short pants', 
    ]

    device = "cuda" #if torch.cuda.is_available() else "cpu"
    model, preprocess = clip.load("ViT-B/32", device=device)

    # image = preprocess(Image.open("CLIP.png")).unsqueeze(0).to(device)
    # text = clip.tokenize(["a diagram", "a dog", "a cat"]).to(device)
    # text = clip.tokenize(cloth_class_text_list).to(device)

    text_upper = clip.tokenize(cloth_class_text_list_upper).to(device)
    text_lower = clip.tokenize(cloth_class_text_list_lower).to(device)
    text_sleeve = clip.tokenize(cloth_class_text_list_upper_sleev).to(device)
    text_blazer_jacket = clip.tokenize(cloth_class_text_list_upper_blazer_jacket).to(device)

    label_dict_ = []
    for i in tqdm(range(len(img_paths))):
        with torch.no_grad():
            image = preprocess(Image.open(os.path.join(img_folder, img_paths[i]))).unsqueeze(0).to(device)
            # image_features = model.encode_image(image)
            # text_features = model.encode_text(text)
            
            # logits_per_image, logits_per_text = model(image, text)
            # probs = logits_per_image.softmax(dim=-1).cpu().numpy()
            logits_per_image, logits_per_text = model(image, text_upper)
            probs_upper = logits_per_image.softmax(dim=-1).cpu().numpy()

            logits_per_image, logits_per_text = model(image, text_lower)
            probs_lower = logits_per_image.softmax(dim=-1).cpu().numpy()

            logits_per_image, logits_per_text = model(image, text_sleeve)
            probs_sleeve = logits_per_image.softmax(dim=-1).cpu().numpy()

            logits_per_image, logits_per_text = model(image, text_blazer_jacket)
            probs_blazer_jacket = logits_per_image.softmax(dim=-1).cpu().numpy()

        # print(probs)
        # print(img_paths[i], '->', probs.argmax(), '->', cloth_class_text_list[probs.argmax()])

        label_dict = {'img_paths': img_paths[i],
            'label_up': str(probs_upper.argmax()),
            'label_low': str(probs_lower.argmax()),
            'label_sleeve': str(probs_sleeve.argmax()),
            'label_balzer_jacket': str(probs_blazer_jacket.argmax()),
            'label_descip_up': cloth_class_text_list_upper[probs_upper.argmax()],
            'label_descip_low': cloth_class_text_list_lower[probs_lower.argmax()],
            'label_descip_sleeve': cloth_class_text_list_upper_sleev[probs_sleeve.argmax()],
            'label_descip_balzer_jacket': cloth_class_text_list_upper_blazer_jacket[probs_blazer_jacket.argmax()]}

        label_dict_.append(label_dict)

    with open(output_lables_json, 'w') as f:
        json.dump(label_dict_, f)

        # print(label_dict)

def clip_labelling_of_zalando_augmented():
    img_folder = '/is/cluster/fast/ssanyal/project_4/data/zalando/clo_img_seg_original_excluding_jacket_blazer' # '/is/cluster/fast/ssanyal/project_4/data/zalando/512x512_with_alpha_samples' # 
    img_paths = os.listdir(img_folder)[:100]

    output_lables_json = '/is/cluster/fast/ssanyal/project_4/data/zalando/clip_labelling/clothing_labels_excluding_jacket_blazer_augmented_maxscore.json'

    # img_paths = ['RB022G002-K11@18.png']

    # cloth_class_text = 'person wearing short shirt and short pants'

    # augmented_texts = [
    #     'a bad photo of {}',
    #     'a good photo of {}',
    #     'a low resolution photo of {}',
    #     'a high resolution photo of {}',
    #     'a no background photo of {}',
    #     'a fashion photo of {}',
    #     'a photo of {}',
    #     'a blurry photo of {}',
    #     # 'a photo of a cool {}',
    #     # 'a photo of a single {}',
    #     'a segmented photo of {}',
    # ]

    augmented_texts = [
        'a bad photo of {}',
        'a good photo of {}',
        'a low resolution photo of {}',
        'a high resolution photo of {}',
        'a no background photo of {}',
        'a fashion photo of {}',
        'a photo of {}',
        'a blurry photo of {}',
        'a photo of a cool {}',
        'a photo of a single {}',
        'a segmented photo of {}',
        'a bright photo of {}',
        'a drawing of {}',
        'a pixalated photo of {}',
        'a painting of {}',
        'a photo of a dirty {}',
        'a photo of a clean {}',
        'a photo of a hard to see {}',
        'a sketch of {}',
        'a sculpture of {}',
        'a rendering of {}',
        'a close up photo of {}',
    ]

    augmented_length = len(augmented_texts)

    # cloth_class_text_list = [
    #     'a person wearing long t-shirt and short pants', 
    #     'a person wearing short t-shirt and short pants', 
    #     'a person wearing long t-shirt and long pants', 
    #     'a person wearing short t-shirt and long pants', 
    #     'a person wearing long shirt and long pants', 
    #     'a person wearing long shirt and short pants', 
    #     'a person wearing short shirt and long pants', 
    #     'a person wearing short shirt and short pants', 
    #     'a person wearing long open blazer and long pants', 
    #     'a person wearing long open blazer and short pants',
    #     'a person wearing long close blazer and long pants',
    #     'a person wearing long close blazer and short pants', 
    #     'a person wearing long open jacket and long pants',
    #     'a person wearing long open jacket and short pants',
    #     'a person wearing long close jacket and long pants',
    #     'a person wearing long close jacket and short pants',
    #     ]

    # cloth_class_text_list = [
    #     'a person wearing long sleev t-shirt and short pants', 
    #     'a person wearing short sleev t-shirt and short pants', 
    #     'a person wearing long sleev t-shirt and long pants', 
    #     'a person wearing short sleev t-shirt and long pants', 
    #     'a person wearing long sleev shirt and long pants', 
    #     'a person wearing long sleev shirt and short pants', 
    #     'a person wearing short sleev shirt and long pants', 
    #     'a person wearing short sleev shirt and short pants', 
    #     'a person wearing long sleev open blazer and long pants', 
    #     'a person wearing long sleev open blazer and short pants',
    #     'a person wearing long sleev close blazer and long pants',
    #     'a person wearing long sleev close blazer and short pants', 
    #     'a person wearing long sleev  open jacket and long pants',
    #     'a person wearing long sleev open jacket and short pants',
    #     'a person wearing long sleev close jacket and long pants',
    #     'a person wearing long sleev close jacket and short pants',
    #     ]

    # cloth_class_text_list_upper = [
    #     'a person is wearing long sleev t-shirt', 
    #     'a person is wearing short sleev t-shirt', 
    #     'a person is wearing long sleev t-shirt', 
    #     'a person is wearing short sleev t-shirt', 
    #     'a person is wearing long sleev shirt', 
    #     'a person is wearing long sleev shirt', 
    #     'a person is wearing short sleev shirt', 
    #     'a person is wearing short sleev shirt', 
    #     'a person is wearing long sleev open blazer', 
    #     'a person is wearing long sleev open blazer',
    #     'a person is wearing long sleev close blazer',
    #     'a person is wearing long sleev close blazer', 
    #     'a person is wearing long sleev  open jacket',
    #     'a person is wearing long sleev open jacket',
    #     'a person is wearing long sleev close jacket',
    #     'a person is wearing long sleev close jacket',
    #     ]

    cloth_class_text_list_upper = [
        'a person wearing a t-shirt', 
        'a person wearing a shirt', 
        # 'a person wearing a blazer',
        # 'a person wearing a jacket',
        ]

    cloth_class_text_list_upper_blazer_jacket = [
        'a upper body clothing which is open', 
        'a upper body clothing which is close', 
        ]

    cloth_class_text_list_upper_sleev = [
        'a person wearing long sleeves', 
        'a person wearing short sleeves', 
        ]

    cloth_class_text_list_lower = [
        'a person wearing long pants', 
        'a person wearing short pants', 
    ]

    cloth_class_text_list_upper_augmented = []
    for j in cloth_class_text_list_upper:
        for i in augmented_texts:
            cloth_class_text_list_upper_augmented.append(i.format(j))

    cloth_class_text_list_upper_blazer_jacket_augmented = []
    for j in cloth_class_text_list_upper_blazer_jacket:
        for i in augmented_texts:
            cloth_class_text_list_upper_blazer_jacket_augmented.append(i.format(j))

    cloth_class_text_list_upper_sleev_augmented = []
    for j in cloth_class_text_list_upper_sleev:
        for i in augmented_texts:
            cloth_class_text_list_upper_sleev_augmented.append(i.format(j))

    cloth_class_text_list_lower_augmented = []
    for j in cloth_class_text_list_lower:
        for i in augmented_texts:
            cloth_class_text_list_lower_augmented.append(i.format(j))

    device = "cuda" #if torch.cuda.is_available() else "cpu"
    model, preprocess = clip.load("ViT-B/32", device=device)

    text_upper = clip.tokenize(cloth_class_text_list_upper_augmented).to(device)
    text_lower = clip.tokenize(cloth_class_text_list_lower_augmented).to(device)
    text_sleeve = clip.tokenize(cloth_class_text_list_upper_sleev_augmented).to(device)
    text_blazer_jacket = clip.tokenize(cloth_class_text_list_upper_blazer_jacket_augmented).to(device)

    label_dict_ = []
    for i in tqdm(range(len(img_paths))):
        with torch.no_grad():
            image = preprocess(Image.open(os.path.join(img_folder, img_paths[i]))).unsqueeze(0).to(device)
            # image_features = model.encode_image(image)
            # text_features = model.encode_text(text)
            
            # logits_per_image, logits_per_text = model(image, text)
            # probs = logits_per_image.softmax(dim=-1).cpu().numpy()
            logits_per_image, logits_per_text = model(image, text_upper)
            probs_upper = logits_per_image.softmax(dim=-1).cpu().numpy()
            # probs_upper = np.array([probs_upper[:, :augmented_length].sum(), 
            #                         probs_upper[:, augmented_length:2*augmented_length].sum(), 
            #                         probs_upper[:, 2*augmented_length:3*augmented_length].sum(), 
            #                         probs_upper[:, 3*augmented_length:].sum()])

            probs_upper = np.array([probs_upper[:, :augmented_length].max(), 
                                    probs_upper[:, augmented_length:2*augmented_length].max()])

            logits_per_image, logits_per_text = model(image, text_lower)
            probs_lower = logits_per_image.softmax(dim=-1).cpu().numpy()
            probs_lower = np.array([probs_lower[:, :augmented_length].max(),
                                    probs_lower[:, augmented_length:].max()])

            logits_per_image, logits_per_text = model(image, text_sleeve)
            probs_sleeve = logits_per_image.softmax(dim=-1).cpu().numpy()
            probs_sleeve = np.array([probs_sleeve[:, :augmented_length].max(),
                                    probs_sleeve[:, augmented_length:].max()])

            logits_per_image, logits_per_text = model(image, text_blazer_jacket)
            probs_blazer_jacket = logits_per_image.softmax(dim=-1).cpu().numpy()
            probs_blazer_jacket = np.array([probs_blazer_jacket[:, :augmented_length].max(),
                                    probs_blazer_jacket[:, augmented_length:].max()])
        # ipdb.set_trace()

        # print(probs)
        # print(img_paths[i], '->', probs.argmax(), '->', cloth_class_text_list[probs.argmax()])

        label_dict = {'img_paths': img_paths[i],
            'label_up': str(probs_upper.argmax()),
            'label_low': str(probs_lower.argmax()),
            'label_sleeve': str(probs_sleeve.argmax()),
            'label_balzer_jacket': str(probs_blazer_jacket.argmax()),
            'label_descip_up': cloth_class_text_list_upper[probs_upper.argmax()],
            'label_descip_low': cloth_class_text_list_lower[probs_lower.argmax()],
            'label_descip_sleeve': cloth_class_text_list_upper_sleev[probs_sleeve.argmax()],
            'label_descip_balzer_jacket': cloth_class_text_list_upper_blazer_jacket[probs_blazer_jacket.argmax()]}

        label_dict_.append(label_dict)

    with open(output_lables_json, 'w') as f:
        json.dump(label_dict_, f)

        # print(label_dict)


def Diffusion_BLIP_labelling_of_zalando_augmented():
    ## This gives almost 100% accuracy for seleccitng the long or short sleeves and long or short pants classification
    from lavis.models import load_model_and_preprocess

    img_folder = '/is/cluster/fast/ssanyal/project_4/data/zalando/clo_img_seg_original_excluding_jacket_blazer' # '/is/cluster/fast/ssanyal/project_4/data/zalando/512x512_with_alpha_samples' # 
    img_paths = os.listdir(img_folder) # os.listdir(img_folder)[:100] # 

    output_lables_json = '/is/cluster/fast/ssanyal/project_4/data/zalando/clip_labelling/clothing_labels_excluding_jacket_blazer_blip_full_16362_color.json'

    device = 'cuda'
    model, vis_processors, txt_processors = load_model_and_preprocess(name="blip_vqa", model_type="vqav2", is_eval=True, device=device)

    # question_clothtype = "Does the upper body clothing is a t-shirt or shirt?"
    # question_sleeves = "Does the upper body clothing has short or long sleeves?"
    # question_pants = "Does the lower body clothing has short or long pants?"

    question_sleeves = "What is the color of the upper body clothing of the person wearing in the image?"
    question_pants = "What is the color of the pants of the person wearing in the image?"

    label_dict_ = []
    for i in tqdm(range(len(img_paths))):

        raw_image = Image.open(os.path.join(img_folder, img_paths[i])).convert("RGB")
        image = vis_processors["eval"](raw_image).unsqueeze(0).to(device)
        # question_clothtype = txt_processors["eval"](question_clothtype)
        question_sleeves = txt_processors["eval"](question_sleeves)
        question_pants = txt_processors["eval"](question_pants)
        # clothtype = model.predict_answers(samples={"image": image, "text_input": question_clothtype}, inference_method="generate")
        sleeves = model.predict_answers(samples={"image": image, "text_input": question_sleeves}, inference_method="generate")
        pants = model.predict_answers(samples={"image": image, "text_input": question_pants}, inference_method="generate")

        label_dict = {'img_paths': img_paths[i],
            # 'label_up': str(probs_upper.argmax()),
            # 'label_low': str(probs_lower.argmax()),
            # 'label_sleeve': str(probs_sleeve.argmax()),
            # 'label_balzer_jacket': str(probs_blazer_jacket.argmax()),
            # 'label_descip_up': clothtype,
            'label_descip_low': pants,
            'label_descip_sleeve': sleeves,}
            # 'label_descip_balzer_jacket': cloth_class_text_list_upper_blazer_jacket[probs_blazer_jacket.argmax()]}
        label_dict_.append(label_dict)

    with open(output_lables_json, 'w') as f:
        json.dump(label_dict_, f)



def check_labelling():
    labels_json = '/is/cluster/fast/ssanyal/project_4/data/zalando/clip_labelling/clothing_labels_excluding_jacket_blazer_CLIP_blip_combined_full.json'
    with open(labels_json, 'r') as f:
        data = json.load(f)

    img_folder = '/is/cluster/work/ssanyal/project_4/data/stylegan3/zalando/RGB_with_alpha/imgs/clo_imgs_seg'
    out_folder = '/is/cluster/fast/ssanyal/project_4/data/zalando/clip_labelling/labelled_img_3_clothing_labels_excluding_jacket_blazer_CLIP_blip_combined_from_100_150'

    if not os.path.exists(out_folder): 
        os.makedirs(out_folder)

    for i in tqdm(range(100, 150)):
        print(i+1)
        img_name = os.path.join(img_folder, data[i]['img_paths'])
        # img_label = data[i]['label_descip']
        # img_label = data[i]['label_descip_up'] + ' and ' + data[i]['label_descip_low']
        img_label = data[i]['label_descip_up'] + ' and ' + data[i]['label_descip_sleeve'] + ' and ' + data[i]['label_descip_low']
        # print(img_label)
        img = plt.imread(img_name)
        plt.imshow(img)
        plt.text(0, 0, data[i]['label_descip_up'], c='r')
        # plt.text(0, 20, data[i]['label_descip_balzer_jacket'], c='r')
        plt.text(0, 20, data[i]['clothing_label'], c='r')
        plt.text(0, 60, 'sleeve type: ' + data[i]['label_descip_sleeve'], c='r')
        plt.text(0, 40, 'pant type: ' + data[i]['label_descip_low'], c='r')
        plt.savefig(os.path.join(out_folder, data[i]['img_paths']))
        plt.close()
        # plt.show()

    # for i in tqdm(range(len(data))):
    #     print(i+1)
    #     img_name = os.path.join(img_folder, data[i]['img_paths'])
    #     # img_label = data[i]['label_descip']
    #     # img_label = data[i]['label_descip_up'] + ' and ' + data[i]['label_descip_low']
    #     img_label = data[i]['label_descip_up'][0] + ' and ' + data[i]['label_descip_sleeve'][0] + ' and ' + data[i]['label_descip_low'][0]
    #     print(img_label)
    #     img = plt.imread(img_name)
    #     plt.imshow(img)
    #     plt.text(0, 0, 'clothtype: ' + data[i]['label_descip_up'][0], c='r')
    #     # plt.text(0, 20, data[i]['label_descip_balzer_jacket'], c='r')
    #     plt.text(0, 60, 'sleeve type: ' + data[i]['label_descip_sleeve'][0], c='r')
    #     plt.text(0, 40, 'pant type: ' + data[i]['label_descip_low'][0], c='r')
    #     plt.savefig(os.path.join(out_folder, data[i]['img_paths']))
    #     plt.close()
    #     # plt.show()

def combine_CLIP_and_BLIP_labels():
    CLIP_labels_path = '/is/cluster/fast/ssanyal/project_4/data/zalando/clip_labelling/clothing_labels_excluding_jacket_blazer_withoitaug_full_16362_CLIP.json'
    BLIP_labels_path = '/is/cluster/fast/ssanyal/project_4/data/zalando/clip_labelling/clothing_labels_excluding_jacket_blazer_blip_full_16362.json'

    output_lables_json = '/is/cluster/fast/ssanyal/project_4/data/zalando/clip_labelling/clothing_labels_excluding_jacket_blazer_CLIP_blip_combined_full.json'
    output_lables_json_2 = '/is/cluster/fast/ssanyal/project_4/data/zalando/clip_labelling/clothing_labels_excluding_jacket_blazer_CLIP_blip_combined_dictionarytype_full.json'

    with open(CLIP_labels_path, 'r') as f:
        CLIP_labels = json.load(f)

    with open(BLIP_labels_path, 'r') as f:
        BLIP_labels = json.load(f)

    label_dict_ = []
    label_dict_2 = {}
    for i in tqdm(range(len(CLIP_labels))):
        # print(CLIP_labels[i]['img_paths'], '->', BLIP_labels[i]['img_paths'])
        # print(BLIP_labels[i]['label_descip_sleeve'])
        # print(BLIP_labels[i]['label_descip_low'])
        upper_body_cloth_type = CLIP_labels[i]['label_descip_up'].split(' ')[-1]
        sleeve_length = BLIP_labels[i]['label_descip_sleeve'][0]
        pant_length = BLIP_labels[i]['label_descip_low'][0]
        assert CLIP_labels[i]['img_paths'] == BLIP_labels[i]['img_paths']
        try:
            assert sleeve_length in ['long', 'short']
        except:
            print('assertion error<------------------------------>label_descip_sleeve: {}->{}'.format(BLIP_labels[i]['img_paths'], sleeve_length))
            sleeve_length = 'long'

        try:
            assert pant_length in ['long', 'short']
        except:
            print('assertion error<------------------------------>label_descip_low: {}->{}'.format(BLIP_labels[i]['img_paths'], pant_length))
            pant_length = 'long'


        # print('{} and the sleeves are {} and the pants are {}'.format(CLIP_labels[i]['label_descip_up'], BLIP_labels[i]['label_descip_sleeve'], BLIP_labels[i]['label_descip_low']))

        # ipdb.set_trace()

        if upper_body_cloth_type == 't-shirt' and sleeve_length == 'short' and pant_length == 'short':
            clothing_label = 'shortshort'
        elif upper_body_cloth_type == 't-shirt' and sleeve_length == 'long' and pant_length == 'short':
            clothing_label = 'longshort'
        elif upper_body_cloth_type == 't-shirt' and sleeve_length == 'short' and pant_length == 'long':
            clothing_label = 'shortlong'
        elif upper_body_cloth_type == 't-shirt' and sleeve_length == 'long' and pant_length == 'long':
            clothing_label = 'longlong'
        elif upper_body_cloth_type == 'shirt' and pant_length == 'long':
            clothing_label = 'shirtlong'
        elif upper_body_cloth_type == 'shirt' and pant_length == 'short':
            clothing_label = 'shirtshort'

        label_dict = {'img_paths': CLIP_labels[i]['img_paths'],
            'clothing_label' : clothing_label,
            'label_descip_up': upper_body_cloth_type,
            'label_descip_low': pant_length,
            'label_descip_sleeve': sleeve_length,}
        label_dict_.append(label_dict)
        label_dict_2[CLIP_labels[i]['img_paths']] = label_dict

    with open(output_lables_json, 'w') as f:
        json.dump(label_dict_, f)

    with open(output_lables_json_2, 'w') as f:
        json.dump(label_dict_2, f)
        


def CLIP_features_from_BLIP_labels():
    BLIP_json_path = '/is/cluster/fast/ssanyal/project_4/data/zalando/clip_labelling/clothing_labels_excluding_jacket_blazer_blip_full_16362_color.json'
    with open(BLIP_json_path, 'r') as f:
        BLIP_labels = json.load(f)

    img_folder = '/is/cluster/work/ssanyal/project_4/data/stylegan3/zalando/RGB_with_alpha/imgs/clo_imgs_seg'

    out_folder = '/is/cluster/fast/ssanyal/project_4/data/zalando/clip_labelling/color_labelling/blip_2_clip'

    if not os.path.exists(out_folder):
        os.makedirs(out_folder)


    device = "cuda" #if torch.cuda.is_available() else "cpu"
    model, preprocess = clip.load("ViT-B/32", device=device)

    text_add_on = ['The color of the upper body clothing is {} and the color of the pants is {}']

    print_labels = False

    if print_labels:
        if not os.path.exists(out_folder + '_png'):
            os.makedirs(out_folder + '_png')

    for i in tqdm(range(len(BLIP_labels))):
        with torch.no_grad():
            img_fn = BLIP_labels[i]['img_paths']
            full_text = text_add_on[0].format(BLIP_labels[i]['label_descip_sleeve'][0], BLIP_labels[i]['label_descip_low'][0])
            # print(full_text)
            text = clip.tokenize(full_text).to(device)

            text_features = model.encode_text(text)

            np.save(os.path.join(out_folder, img_fn[:-3]+'npy'), text_features[0].cpu().numpy().astype(np.float32))

            if print_labels:
                img = plt.imread(os.path.join(img_folder, img_fn))[:,:,:3]
                plt.imshow(img)
                plt.text(0, 0, full_text, c='r')
                plt.text(0, 20, BLIP_labels[i]['label_descip_low'][0], c='r')
                plt.savefig(os.path.join(out_folder + '_png', img_fn))
                plt.close()


        # ipdb.set_trace()


if __name__ == "__main__":
    # clip_labelling_of_zalando()
    # clip_labelling_of_zalando_augmented()
    # Diffusion_BLIP_labelling_of_zalando_augmented()
    # check_labelling()
    # combine_CLIP_and_BLIP_labels()
    CLIP_features_from_BLIP_labels()
